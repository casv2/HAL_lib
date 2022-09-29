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

def assemble_lsq(ace_basis, atoms_list):
    num_obs = np.sum([1 + len(at)*3 for at in atoms_list])
    len_ace_basis = Main.eval("length(B)")

    Psi = np.zeros((num_obs, len_ace_basis))
    Y = np.zeros(num_obs)

    i = 0
    for at in atoms_list:
        Psi[i,:] = np.array(energy(ace_basis, convert(ASEAtoms(at)))).flatten()
        Y[i] = np.array(at.info["energy"]).flatten()
        i += 1

        Frows = len(at)*3
        Psi[i:i+Frows, :] = np.reshape(np.array(forces(ace_basis, convert(ASEAtoms(at)))).flatten(), (len_ace_basis, Frows)).transpose()
        Y[i:i+Frows] = np.array(at.arrays["forces"]).flatten()
        i += Frows

        # Vrows = 6
        # Psi[i:i+Vrows, :] = np.reshape(np.array(virial(ace_basis, convert(ASEAtoms(at)))).flatten(), (len_ace_basis, Vrows)).transpose()
        # Y[i:i+Vrows] = np.array(at.arrays["virial"]).flatten()

    return Psi, Y

def fit(E0s, B, atoms_list, solver, ncomms=32):
    Psi, Y = assemble_lsq(B, atoms_list)

    solver.fit(Psi, Y)
    c = solver.coef_
    sigma = solver.sigma_
    #min_sigma_eig_val = np.min(np.isreal(np.linalg.eigvals(sigma)))

    comms = np.random.multivariate_normal(c, sigma, size=ncomms)
    IP = ace_basis.combine(E0s, B, c, comms)
    
    return IP



