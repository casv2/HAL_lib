#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
from HAL_lib import ace_basis
Main.eval("using ASE, JuLIP, ACE1")

import numpy as np

from julia.JuLIP import energy, forces, stress
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")
ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")


def assemble_lsq(ace_basis, atoms_list):
    num_obs = np.sum([len(at)*3 for at in atoms_list])
    len_ace_basis = Main.eval("length(B)")

    Psi = np.zeros((num_obs, len_ace_basis))
    Y = np.zeros(num_obs)

    i = 0
    for at in atoms_list:
        rows = len(at)*3
        Psi[i:i+rows, :] = np.reshape(np.array(forces(ace_basis, convert(ASEAtoms(at)))).flatten(), (len_ace_basis, len(at)*3)).transpose()
        Y[i:i+rows] = np.array(at.arrays["forces"]).flatten()
        i += rows

    return Psi, Y

def fit(ref_pot, B, atoms_list, solver, ncomms=32):
    Psi, Y = assemble_lsq(B, atoms_list)

    solver.fit(Psi, Y)
    c = solver.coef_
    sigma = solver.sigma_
    min_sigma_eig_val = np.min(np.isreal(np.linalg.eigvals(sigma)))

    comms = np.random.multivariate_normal(c, sigma + min_sigma_eig_val*np.eye(len(c), dtype=float), size=ncomms)
    IP = ace_basis.combine(ref_pot, B, c, comms)
    
    return IP



