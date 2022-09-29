from copy import deepcopy
import numpy as np

from HAL_lib import lsq
from HAL_lib import ace_basis
from HAL_lib import MD
from HAL_lib import com

from ase.units import fs
from ase.units import kB
from ase.units import GPa

import matplotlib.pyplot as plt

def HAL(E0s, basis_info, run_info, atoms_list, start_configs, solver): #calculator
    #general settings
    niters = run_info["niters"]
    ncomms = run_info["ncomms"]
    nsteps = run_info["nsteps"]
    tau_rel = run_info["tau_rel"]
    dt = run_info["dt"]

    #
    baro_settings = { "baro" : False}
    thermo_settings = { "thermo" : False}

    #baro/thermo on or not
    if run_info["baro"] == True:
        baro_settings["baro"] = True
        baro_settings["P"] = run_info["P"]
        baro_settings["mu"] = run_info["mu"]
    if run_info["thermo"] == True:
        thermo_settings["thermo"] = True
        thermo_settings["T"] = run_info["T"]
        thermo_settings["gamma"] = run_info["gamma"]
    # if run_info["swap"] == True:
    #     swap_step = run_info["swap_step"]
    # if run_info["vol"] == True:
    #     vol_step = run_info["vol_step"]
    
    for (j, start_config) in enumerate(start_configs):
        for i in range(niters):
            init_config = deepcopy(start_config)
            m = j*niters + i

            B = ace_basis.full_basis(basis_info);
            IP, IPs = lsq.fit(E0s, B, atoms_list, solver, ncomms=ncomms)
            
            #here we fill in the keywords for run
            E_tot, E_kin, E_pot, T_s, P_s, f_s =  run(IP, IPs, init_config, nsteps, dt, tau_rel, baro_settings, thermo_settings)

            plot(E_tot, E_kin, E_pot, T_s, P_s, f_s, m)


def run(IP, IPs, at, nsteps, dt, tau_rel, baro_settings, thermo_settings):
    E_tot = np.zeros(nsteps)
    E_pot = np.zeros(nsteps)
    E_kin = np.zeros(nsteps)
    T_s = np.zeros(nsteps)
    P_s = np.zeros(nsteps)
    f_s = np.zeros(nsteps)

    m_F_bar = np.zeros(nsteps)
    m_F_bias = np.zeros(nsteps)

    at.set_calculator(IP)
    E0 = at.get_potential_energy()

    running=True
    i=0

    tau=0.0
    while running and i < nsteps:
        at, F_bar_mean, F_bias_mean = MD.Velocity_Verlet(IP, IPs, at, dt * fs, tau, baro_settings=baro_settings, thermo_settings=thermo_settings)
        
        m_F_bar[i] = F_bar_mean
        m_F_bias[i] = F_bias_mean

        if i > 100:
            tau = (tau_rel * np.mean(m_F_bar[i-99:i])) / np.mean(m_F_bias[i-99:i])
        else:
            tau = 0.0

        print(tau)

        E_kin[i] = at.get_kinetic_energy()/len(at)
        E_pot[i] = (at.get_potential_energy() - E0)/len(at)
        E_tot[i] = E_kin[i] + E_pot[i]
        T_s[i] = (at.get_kinetic_energy()/len(at)) / (1.5 * kB)
        P_s[i] = (np.trace(at.get_stress(voigt=False))/3) / GPa
        f_s[i] = com.get_fi(IP, IPs, at)

        i += 1

        if i > nsteps:
            running=False

    return E_tot[:i], E_kin[:i], E_pot[:i], T_s[:i], P_s[:i], f_s[:i]
    


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
    plt.savefig("./plot_{}.pdf".format(m))

    