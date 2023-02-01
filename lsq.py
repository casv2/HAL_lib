#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main
from HAL_lib import ace_basis
Main.eval("using ASE, JuLIP, ACE1")

import numpy as np

from ase.atoms import Atoms

from julia.JuLIP import energy, forces, virial
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")
ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")

def lsq_section(B, E0s, at, data_keys, weights, Fmax=1e32):
    Psi = []
    Y = []

    # len_B = Main.eval("length(B)")

    if data_keys["E"] in at.info:
        # N_B
        E_B = np.array(energy(B, convert(ASEAtoms(at))))
        Psi.append(weights["E"] * E_B)
        Y.append(weights["E"] * (at.info[data_keys["E"]] - np.sum([at.get_chemical_symbols().count(EL) * E0 for EL, E0 in E0s.items()])))

    if data_keys["F"] in at.arrays:
        # N_B x N_atoms x 3
        F_B = np.array(forces(B, convert(ASEAtoms(at))))

        # filter F <= Fmax
        F = at.arrays[data_keys["F"]]
        F_filter = np.linalg.norm(F, axis=1) <= Fmax
        F = F[F_filter, :]

        # N_B x (N_atoms with not too large F) x 3
        F_B = F_B[:, F_filter, :]

        # N_B x (N_atoms with not too large F) * 3
        F_B = F_B.reshape((F_B.shape[0], -1))

        Psi.extend(weights["F"] * F_B.T)
        Y.extend(weights["F"] * F.reshape((-1)))

    if data_keys["V"] in at.info:
        # N_B x 3 x 3
        V_B = np.array(virial(B, convert(ASEAtoms(at))))

        # select 6 independent elements of V (standard Voigt order)
        # note that V might come in as (9,) or (3,3), so flatten first
        V = np.array(at.info[data_keys["V"]]).reshape((-1))
        Vi = [0, 1, 2, 0, 1, 2]
        Vj = [0, 1, 2, 1, 2, 0]
        V = V[np.arange(9).reshape((3,3))[Vi, Vj]]

        # N_B x 6
        V_B = V_B[:, Vi, Vj]

        Psi.extend(weights["V"] * V_B.T)
        Y.extend(weights["V"] * V)

    return Psi, Y

def add_lsq(B, E0s, atoms_list, data_keys, weights, Fmax, Psi=None, Y=None):
    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]

    Psi_new = []
    Y_new = []
    for at in atoms_list:
        Psi_sec, Y_sec = lsq_section(B, E0s, at, data_keys, weights, Fmax)
        Psi_new.extend(Psi_sec)
        Y_new.extend(Y_sec)

    if Psi is None:
        assert Y is None
        Psi = np.array(Psi_new)
        Y = np.array(Y_new)
    else:
        assert Y is not None
        Psi = np.append(Psi, np.array(Psi_new), axis=0)
        Y = np.append(Y, np.array(Y_new))

    return Psi, Y

def fit(Psi, Y, B, E0s, solver, ncomms=32, mvn_hermitian=True):
    solver.fit(Psi, Y)
    c = solver.coef_
    sigma = solver.sigma_
    score = solver.scores_[-1]

    # Code below is for ARD solver, TO DO:
    if sigma.shape[0] != len(c):
        sigma = np.zeros((len(c), len(c)), dtype=float)
        for (i,non_zero) in enumerate(np.nonzero(c)):
            coeff = solver.sigma_[i, i]
            sigma[non_zero, non_zero] = coeff
        #comms = np.random.multivariate_normal(c, 0.5*(sigma_large + sigma_large.T), size=ncomms)
    #else:

    #sigma_min_eig_val = np.min(np.real(np.linalg.eigvals(sigma)))
    #sigma_reg = sigma + (np.eye(sigma.shape[0]) * np.abs(sigma_min_eig_val) * 10)
    #sigma_reg_min_eig_val = np.min(np.real(np.linalg.eigvals(sigma_reg)))

    print("score: {}".format(score))

    #if mvn_hermitian:
    #    comms = multivariate_normal_hermitian(c, sigma, size=ncomms)
    #else:
    comms = np.random.multivariate_normal(c, sigma, size=ncomms)
    
    IP, IPs = ace_basis.combine(B, c, E0s, comms)
    
    return IP, IPs
