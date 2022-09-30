import numpy as np

#from copy import deepcopy
from ase.units import kB

## swap step
def MC_swap_step(IP, IPs, at, tau, temp):
    E1 = get_HAL_E(IP, IPs, at, tau)

    els = at.get_chemical_symbols()
    ms = at.get_masses()

    found = False
    while found == False:
        i1, i2 = np.random.randint(len(at), size=2) 

        el1, el2 = els[i1], els[i2]

        if el1 != el2:
            found = True

    
    m1, m2 = ms[i1], ms[i2]
    el1, el2 = els[i1], els[i2]

    at[i1].symbol = el2
    at[i1].mass = m2

    at[i2].symbol = el1
    at[i2].mass = m1

    E2 = get_HAL_E(IP, IPs, at, tau)

    p = np.exp((E1 - E2) / (kB * temp))

    if np.random.rand() < p:
        print("ACCEPT_SWAP")
        at[i1].symbol = el2
        at[i1].mass = m2

        at[i2].symbol = el1
        at[i2].mass = m1

        return at
    else:
        at[i1].symbol = el1
        at[i1].mass = m1

        at[i2].symbol = el2
        at[i2].mass = m2

        return at


## vol step
def MC_vol_step(IP, IPs, at, tau, temp):
    C1 = at.cell
    C2 = at.cell + np.random.normal(0.0, 0.05, size=(3, 3))

    E1 = get_HAL_E(IP, IPs, at, tau)

    at.set_cell(C2, scale_atoms=True)

    E2 = get_HAL_E(IP, IPs, at, tau)

    p = np.exp((E1 - E2) / (kB * temp))

    if np.random.rand() < p:
        print("ACCEPT_VOL")
        at.set_cell(C2, scale_atoms=True)
        return at
    else:
        at.set_cell(C1, scale_atoms=True)
        return at

def get_HAL_E(IP, IPs, at, tau):
    at.set_calculator(IP)

    E_bar = at.get_potential_energy()
    E_comms = []

    for IP_com in IPs:
        at.set_calculator(IP_com)
        E_comms.append(at.get_potential_energy())

    E_std = np.sqrt(1/len(IPs) * np.sum([ np.power((E_bar - E_comms[i]), 2) for i in range(len(IPs))]) )

    return E_bar - tau * E_std