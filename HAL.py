from copy import deepcopy
import numpy as np

from HAL_lib import lsq
from HAL_lib import ace_basis
from HAL_lib import MD
from HAL_lib import com
from HAL_lib import MC
from HAL_lib import utils
from HAL_lib import errors

from ase.units import fs
from ase.units import kB
from ase.units import GPa

from ase.io import write

import matplotlib.pyplot as plt

def HAL(B, E0s, weights, run_info, atoms_list, start_configs, solver, calculator=None): #calculator
    #general settings
    niters = run_info["niters"]
    ncomms = run_info["ncomms"]
    nsteps = run_info["nsteps"]
    tau_rel = run_info["tau_rel"]
    tau_hist = run_info["tau_hist"]
    dt = run_info["dt"]
    f_tol = run_info["f_tol"]
    softmax = run_info["softmax"]

    #
    baro_settings = { "baro" : False}
    thermo_settings = { "thermo" : False}
    swap_settings = { "swap" : False}
    vol_settings = { "vol" : False}

    #baro/thermo on or not
    if run_info["baro"] == True:
        baro_settings["baro"] = True
        baro_settings["target_pressure"] = run_info["P"]
        baro_settings["mu"] = run_info["mu"]
    if run_info["thermo"] == True:
        thermo_settings["thermo"] = True
        thermo_settings["T"] = run_info["T"]
        thermo_settings["gamma"] = run_info["gamma"]
    if run_info["swap"] == True:
        swap_settings["swap"] = True
        swap_settings["swap_step"] = run_info["swap_step"]
    if run_info["vol"] == True:
        vol_settings["vol"] = True
        vol_settings["vol_step"] = run_info["vol_step"]
    
    for (j, start_config) in enumerate(start_configs):
        for i in range(niters):
            start_config.calc = None
            current_config = deepcopy(start_config)
            m = j*niters + i

            #B = ace_basis.full_basis(basis_info);

            if m == 0:
                Psi, Y = lsq.assemble_lsq(B, E0s, atoms_list, weights)
            else:
                Psi, Y = lsq.add_lsq(B, E0s, at, weights, Psi, Y)

            ACE_IP, HAL_IP = lsq.fit(Psi, Y, B, E0s, solver, ncomms=ncomms)

            errors.print_errors(ACE_IP, atoms_list)
            
            E_tot, E_kin, E_pot, T_s, P_s, f_s, at = run(ACE_IP, HAL_IP, current_config, nsteps, dt, tau_rel, f_tol, baro_settings, thermo_settings, swap_settings, vol_settings, tau_hist=tau_hist, softmax=softmax)

            plot(E_tot, E_kin, E_pot, T_s, P_s, f_s, m)
            utils.save_pot("HAL_it{}.json".format(m))

            if calculator != None:
                at.set_calculator(calculator)
                at.info["energy"] = at.get_potential_energy()
                at.arrays["forces"] = at.get_forces()
                at.info["virial"] = -1.0 * at.get_volume() * at.get_stress(voigt=False)
                
            del at.arrays["momenta"]

            write("HAL_it{}.extxyz".format(m), at)

            atoms_list.append(at)
    
    return atoms_list

def run(ACE_IP, HAL_IP, at, nsteps, dt, tau_rel, f_tol, baro_settings, thermo_settings, swap_settings, vol_settings, tau_hist=100, softmax=True):
    E_tot = np.zeros(nsteps)
    E_pot = np.zeros(nsteps)
    E_kin = np.zeros(nsteps)
    T_s = np.zeros(nsteps)
    P_s = np.zeros(nsteps)
    f_s = np.zeros(nsteps)

    m_F_bar = np.zeros(nsteps)
    m_F_bias = np.zeros(nsteps)

    at.set_calculator(ACE_IP)
    E0 = at.get_potential_energy()

    running=True
    i=0

    tau=0.0
    while running and i < nsteps:
        at, F_bar_mean, F_bias_mean = MD.Velocity_Verlet(ACE_IP, HAL_IP, at, dt * fs, tau, baro_settings=baro_settings, thermo_settings=thermo_settings)
        
        m_F_bar[i] = F_bar_mean
        m_F_bias[i] = F_bias_mean

        if i > tau_hist:
            tau = (tau_rel * np.mean(m_F_bar[i-tau_hist:i])) / np.mean(m_F_bias[i-tau_hist:i])
        else:
            tau = 0.0

        if (vol_settings["vol"] == True) and (i % vol_settings["vol_step"] == 0):
            at = MC.MC_vol_step(HAL_IP, at, tau, thermo_settings["T"] * kB)

        if (swap_settings["swap"] == True) and (i % swap_settings["swap_step"] == 0):
            at = MC.MC_swap_step(HAL_IP, at, tau, thermo_settings["T"] * kB)

        at.set_calculator(ACE_IP)
        E_kin[i] = at.get_kinetic_energy()/len(at)
        E_pot[i] = (at.get_potential_energy() - E0)/len(at)
        E_tot[i] = E_kin[i] + E_pot[i]
        T_s[i] = (at.get_kinetic_energy()/len(at)) / (1.5 * kB)
        P_s[i] = -1.0 * (np.trace(at.get_stress(voigt=False))/3) / GPa
        f_s[i] = com.get_fi(HAL_IP, at, softmax=softmax)

        if i > nsteps or f_s[i] > f_tol:
            running=False

        if (i % 100):
            print("MD iteration: {}, tau: {}".format(i, tau))

        i += 1

    return E_tot[:i], E_kin[:i], E_pot[:i], T_s[:i], P_s[:i], f_s[:i], at


def plot(E_tot, E_kin, E_pot, T_s, P_s, f_s, m):
    fig, axes = plt.subplots(figsize=(5,8), ncols=1, nrows=4)
    axes[0].plot(E_tot, label="E_tot")
    axes[0].plot(E_kin, label="E_kin")
    axes[0].plot(E_pot, label="E_pot")
    axes[1].plot(T_s)
    axes[2].plot(P_s)
    axes[3].plot(f_s)
    axes[0].set_ylabel("E [ev/atom]")
    axes[1].set_ylabel("T [K]")
    axes[2].set_ylabel("P [GPa]")
    axes[3].set_ylabel("max f_i")
    axes[3].set_xlabel("HAL steps")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./plot_{}.pdf".format(m))

    
