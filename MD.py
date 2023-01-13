import numpy as np

from ase.units import kB
from ase.units import GPa

def random_p_update(p,masses,gamma,kBT,dt):
    v = p / masses
    R = np.random.standard_normal(size=(len(masses), 3))
    c1 = np.exp(-gamma*dt)
    c2 = np.sqrt(1-c1*c1)*np.sqrt(kBT / masses)
    v_new = c1*v + (R* c2)
    return v_new * masses

def barostat(ACE_IP, at, mu, target_pressure):
    at.set_calculator(ACE_IP)
    pressure = -(np.trace(at.get_stress(voigt=False))/3) / GPa
    scale = (1.0 - (mu * (target_pressure - pressure)))
    at.set_cell(at.cell * scale, scale_atoms=True)
    return at

def VelocityVerlet(ACE_IP, CO_IP, at, dt, tau, baro_settings, thermo_settings):
    if "HAL_forces" in at.arrays:
        forces = at.arrays["HAL_forces"]
    else:
        F_bar, F_bias, F_bar_norms, F_bias_norms, dFn  = CO_IP.get_property('force_data',  at) 
        forces = np.asarray(F_bar) - tau * np.asarray(F_bias)

    p = at.get_momenta() + 0.5 * dt * forces
    masses = at.get_masses()[:, np.newaxis]

    if thermo_settings["thermo"] == True: 
        p = random_p_update(p, masses, thermo_settings["gamma"], thermo_settings["T"] * kB, dt)
    at.set_momenta(p, apply_constraint=False)

    r = at.get_positions()
    at.set_positions(r + dt * p / masses)

    F_bar, F_bias, F_bar_norms, F_bias_norms, dFn  = CO_IP.get_property('force_data',  at) 
    forces = np.asarray(F_bar) - tau * np.asarray(F_bias)
    at.arrays["HAL_forces"] = forces

    p = at.get_momenta() + 0.5 * dt * forces

    if baro_settings["baro"] == True:
        at = barostat(ACE_IP, at, baro_settings["mu"], baro_settings["target_pressure"])

    return at, F_bar_norms, F_bias_norms, dFn
