#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main
from HAL_lib import ace_basis
Main.eval("using ASE, JuLIP, ACE1")

import numpy as np

from julia.JuLIP import energy, forces, virial
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")
ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")

def assemble_lsq(B, E0s, atoms_list, weights):
    num_obs = np.sum([1 + len(at)*3 + 9 for at in atoms_list])
    len_B = Main.eval("length(B)")

    Psi = np.zeros((num_obs, len_B))
    Y = np.zeros(num_obs)

    i = 0
    for at in atoms_list:
        Psi[i,:] = weights["E"] * np.array(energy(B, convert(ASEAtoms(at)))).flatten() 
        Y[i] = np.array(at.info["energy"]).flatten() - np.sum([at.get_chemical_symbols().count(EL) * E0 for EL, E0 in E0s.items()])
        i += 1

        Frows = len(at)*3
        Psi[i:i+Frows, :] = weights["F"] * np.reshape(np.array(forces(B, convert(ASEAtoms(at)))).flatten(), (len_B, Frows)).transpose()
        Y[i:i+Frows] = np.array(at.arrays["forces"]).flatten()
        i += Frows

        Vrows = 9
        Psi[i:i+Vrows, :] = weights["V"] * np.reshape(np.array(virial(B, convert(ASEAtoms(at)))).flatten(), (len_B, Vrows)).transpose()
        Y[i:i+Vrows] = np.array(at.info["virial"]).flatten()

    return Psi, Y

def fit(B, E0s, atoms_list, weights, solver, ncomms=32):
    Psi, Y = assemble_lsq(B, E0s, atoms_list, weights)

    solver.fit(Psi, Y)
    c = solver.coef_
    sigma = solver.sigma_

    comms = np.random.multivariate_normal(c, sigma, size=ncomms)
    IP = ace_basis.combine(B, c, E0s, comms)
    
    return IP



