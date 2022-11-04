from sklearn.linear_model import BayesianRidge, ARDRegression
from ase.calculators.castep import Castep
from HAL_lib import HAL
from HAL_lib import ace_basis
from ase.io import read, write

al = read("../init.xyz", ":")
start_configs = read("../init.xyz", ":")


calculator = Castep()
calculator._directory="./_CASTEP"
calculator.param.cut_off_energy=300
calculator.param.mixing_scheme='Pulay'
calculator.param.write_checkpoint='none'
calculator.param.smearing_width=0.1
calculator.param.finite_basis_corr='automatic'
calculator.param.calculate_stress=True
calculator.param.max_scf_cycles=250
calculator.cell.kpoints_mp_spacing=0.04

E0s = { "Ta" : -8431.3011, "W" : -9248.0405 }

weights = { "E" : 15.0, "F" : 1.0 ,"V": 1.0 }

run_info = {
    "niters" : 50,
    "ncomms" : 32,
    "nsteps" : 1000,
    "dt" : 1.0,

    "tau_rel" : 0.05,
    "tau_hist" : 100,

    "softmax" : True,
    "f_tol" : 0.2,
    "eps" : 0.2,

    "thermo" : True,
    "T" : 800,
    "gamma" : 20.0,

    "swap" : True,
    "swap_step" : 50,

    "vol" : True,
    "vol_step" : 50,

    "baro" : False,
    "P" : 10.0,
    "mu" : 1e-4,
}

data_keys = { "E" : "energy", "F" : "forces", "V" : "virial", "Fmax" : 20.0 }


#######

#basis_info = {
#    "elements" : ["W", "Ta"],
#    "cor_order" : 2,
#    "poly_deg_ACE" : 16,
#    "poly_deg_pair" : 16,
#    "r0" : 2.8,
#    "r_in" : 0.5,
#    "r_cut" : 5.5 }

#B = ace_basis.full_basis(basis_info);

#######

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

B = Main.eval("""
            using ACE1: transformed_jacobi, transformed_jacobi_env

            Bsite = rpi_basis(species = [:W, :Ta],
                                N = 2,       # correlation order = body-order - 1
                                maxdeg = 16,  # polynomial degree
                                r0 = 2.8,     # estimate for NN distance
                                rin = 0.5,
                                rcut = 5.5,   # domain for radial basis (cf documentation)
                                pin = 2)                     # require smooth inner cutoff

            trans_r = AgnesiTransform(; r0=2.8, p = 2)
            envelope_r = ACE1.PolyEnvelope(2, 2.8, 6.5)
            Jnew = transformed_jacobi_env(12, trans_r, envelope_r, 6.5)

            Bpair = PolyPairBasis(Jnew, [:W, :Ta])

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
            """)
####

HAL.HAL(B, E0s, weights, run_info, al, data_keys, start_configs, BayesianRidge(), calculator=calculator)
