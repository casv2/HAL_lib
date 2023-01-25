import sys
from copy import deepcopy
import numpy as np
import time

from HAL_lib import lsq
from HAL_lib import MD
from HAL_lib import MC
from HAL_lib import utils
from HAL_lib import errors
from HAL_lib import BO_optim
from HAL_lib import ace_basis

from ase.io import write

from ase.units import fs
from ase.units import kB
from ase.units import GPa

import matplotlib.pyplot as plt

def add_and_fit(B, E0s, data_keys, weights, solver, ncomms, eps, iter_i, new_configs, atoms_list=None, Psi=None, Y=None, mvn_hermitian=True, save=True):
    #assert sum([atoms_list is None, Psi is None, Y is None]) in [0, 3]

    # append configs
    if atoms_list is not None:
        atoms_list.extend(new_configs)
    else:
        atoms_list = list(new_configs)

    print("len atoms_list {}".format(len(atoms_list)))

    # append blocks to design matrix and fitting targets
    if len(new_configs) > 0:
        Psi, Y = lsq.add_lsq(B, E0s, new_configs, data_keys, weights, data_keys.get('Fmax'), Psi, Y)

    if save:
        np.save(f"Psi_it{iter_i}.npy", Psi)
        np.save(f"Y_it{iter_i}.npy", Y)

    t0 = time.time()
    ACE_IP, CO_IP = lsq.fit(Psi, Y, B, E0s, solver, ncomms=ncomms, mvn_hermitian=mvn_hermitian)
    print("TIMING fit", time.time() - t0)
    t0 = time.time()

    utils.save_pot(f"HAL_it{iter_i}.json")

    t0 = time.time()
    errors.print_errors(ACE_IP, atoms_list, data_keys, CO_IP, eps)
    print("TIMING errors", time.time() - t0)
    t0 = time.time()

    sys.stdout.flush()

    return ACE_IP, CO_IP, atoms_list, Psi, Y

def quick_fit(B, E0s, data_keys, weights, solver, ncomms, eps, iter_i, atoms_list, mvn_hermitian=True, save=True):
    Psi, Y = lsq.add_lsq(B, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))

    if save:
        np.save(f"Psi_it{iter_i}.npy", Psi)
        np.save(f"Y_it{iter_i}.npy", Y)

    t0 = time.time()
    ACE_IP, CO_IP = lsq.fit(Psi, Y, B, E0s, solver, ncomms=ncomms, mvn_hermitian=mvn_hermitian)
    print("TIMING fit", time.time() - t0)
    t0 = time.time()

    utils.save_pot(f"HAL_it{iter_i}.json")

    t0 = time.time()
    errors.print_errors(ACE_IP, atoms_list, data_keys, CO_IP, eps)
    print("TIMING errors", time.time() - t0)
    t0 = time.time()

    sys.stdout.flush()

    return ACE_IP, CO_IP, atoms_list

def HAL(optim_basis_param, E0s, weights, run_info, atoms_list, data_keys, start_configs, solver, calculator=None, save=False): #calculator
    niters = run_info["niters"]
    ncomms = run_info["ncomms"]
    nsteps = run_info["nsteps"]
    tau_rel = run_info["tau_rel"]
    tau_hist = run_info["tau_hist"]
    dt = run_info["dt"]
    tol = run_info["tol"]
    eps = run_info["eps"]
    softmax = run_info["softmax"]

    baro_settings = { "baro" : False}
    thermo_settings = { "thermo" : False}
    swap_settings = { "swap" : False}
    vol_settings = { "vol" : False}

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
  
    basis_info = {
        "elements" : optim_basis_param["elements"],     
        "poly_deg_pair" : 22,
        "r_cut_pair" : 7.0,
        "r_0" : 2.5,
        "r_in" : 1.8,
        "r_cut_ACE" : 5.0}

    max_deg_D = {}

    for cor_order in range(2,5):
        for deg in range(3,22):
            basis_info["cor_order"] = cor_order
            basis_info["maxdeg"] = deg
            _, len_B = ace_basis.full_basis(basis_info, return_length=True) 
            if len_B > optim_basis_param["max_len_B"]: #for the max pair
                max_deg_D[cor_order] = deg - 1
                break

    optim_basis_param["max_deg_D"] = max_deg_D

    print("MAX DEG D")
    print(max_deg_D)

    for (j, start_config) in enumerate(start_configs):
        print(f"HAL start_config {j}")
        for i in range(niters):
            start_config.calc = None
            current_config = deepcopy(start_config)
            m = j*niters + i

            if m % optim_basis_param["n_optim"] == 0 or m == 0:
                D = BO_optim.BO_basis_optim(optim_basis_param, solver, atoms_list, E0s, data_keys, weights, D_prior=None)
                B = ace_basis.full_basis(D) 
                ACE_IP, CO_IP, atoms_list = quick_fit(B, E0s, data_keys, weights, solver, ncomms, eps, m, atoms_list, save=save)
            else:
                ACE_IP, CO_IP, atoms_list = quick_fit(B, E0s, data_keys, weights, solver, ncomms, eps, m, atoms_list, save=save)
            
            utils.plot_dimer(ACE_IP, optim_basis_param["elements"], E0s, m=m)

            t0 = time.time()
            E_tot, E_kin, E_pot, T_s, P_s, f_s, at = run(ACE_IP, CO_IP, current_config, nsteps, dt, tau_rel, tol, eps,
                                                         baro_settings, thermo_settings, swap_settings, vol_settings,
                                                         tau_hist=tau_hist, softmax=softmax)

            print("TIMING run", time.time() - t0)
            t0 = time.time()

            plot(E_tot, E_kin, E_pot, T_s, P_s, f_s, tol, m)

            del at.arrays["momenta"]
            del at.arrays["HAL_forces"]

            if calculator != None:
                at.calc = calculator
                at.info[data_keys["E"]] = at.get_potential_energy()
                at.arrays[data_keys["F"]] = at.get_forces()
                try:
                    at.info[data_keys["V"]] = -1.0 * at.get_volume() * at.get_stress(voigt=False)
                except:
                    pass

            print("TIMING reference calculation", time.time() - t0)
            t0 = time.time()

            at.info["config_type"] = "HAL_" + at.info["config_type"]
            write(f"HAL_it{m}.extxyz", at)
            atoms_list += [at]

    return atoms_list

def softmax_func(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run(ACE_IP, CO_IP, at, nsteps, dt, tau_rel, tol, eps, baro_settings, thermo_settings, swap_settings, vol_settings, tau_hist=100, softmax=True):
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
        at, F_bar_norms, F_bias_norms, dFn = MD.VelocityVerlet(ACE_IP, CO_IP, at, dt * fs, tau, baro_settings=baro_settings, thermo_settings=thermo_settings)

        m_F_bar[i] = np.mean(F_bar_norms)
        m_F_bias[i] = np.mean(F_bias_norms)

        if i > tau_hist:
            tau = (tau_rel * np.mean(m_F_bar[i-tau_hist:i])) / np.mean(m_F_bias[i-tau_hist:i])
        else:
            tau = 0.0

        if (vol_settings["vol"] == True) and (i % vol_settings["vol_step"] == 0) and i > 0:
            at = MC.MC_vol_step(CO_IP, at, tau, thermo_settings["T"] * kB)

        if (swap_settings["swap"] == True) and (i % swap_settings["swap_step"] == 0) and i > 0:
            at = MC.MC_swap_step(CO_IP, at, tau, thermo_settings["T"] * kB)

        at.set_calculator(ACE_IP)
        E_kin[i] = at.get_kinetic_energy()/len(at)
        E_pot[i] = (at.get_potential_energy() - E0)/len(at)
        E_tot[i] = E_kin[i] + E_pot[i]
        T_s[i] = (at.get_kinetic_energy()/len(at)) / (1.5 * kB)
        P_s[i] = -1.0 * (np.trace(at.get_stress(voigt=False))/3) / GPa

        p = dFn / (F_bar_norms + eps)

        if softmax:
            f_s[i] = np.max(softmax_func(p))
        else:
            f_s[i] = np.max(p)

        if f_s[i] > tol:
            at.info["HAL_trigger"] = f"force_tol_{f_s[i]}_iter_{i}"
            running=False

        print(f"HAL iteration: {i}, tau: {tau}, max f_i {f_s[i]}")
        sys.stdout.flush()

        i += 1

    if "HAL_trigger" not in at.info:
        at.info["HAL_trigger"] = f"finished_iter_{i}"

    return E_tot[:i], E_kin[:i], E_pot[:i], T_s[:i], P_s[:i], f_s[:i], at


def plot(E_tot, E_kin, E_pot, T_s, P_s, f_s, tol, m):
    fig, axes = plt.subplots(figsize=(5,8), ncols=1, nrows=4)
    axes[0].plot(E_tot, label="E_tot")
    axes[0].plot(E_kin, label="E_kin")
    axes[0].plot(E_pot, label="E_pot")
    axes[1].plot(T_s)
    axes[2].plot(P_s)
    axes[3].plot(f_s)
    axes[3].axhline(y=tol, color="red", label="tol")
    axes[0].set_ylabel("E [ev/atom]")
    axes[1].set_ylabel("T [K]")
    axes[2].set_ylabel("P [GPa]")
    axes[3].set_ylabel("max relative uncertainty")
    axes[3].set_xlabel("HAL steps")
    axes[0].legend(loc="upper left")
    axes[3].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"plot_{m}.pdf")


