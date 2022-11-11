from sklearn.linear_model import BayesianRidge, ARDRegression
from ase.calculators.castep import Castep
from HAL_lib import HAL
from HAL_lib import ace_basis
from ase.io import read, write


###################################
# Reading in initial database (`al`), formed of 1-10 configurations 
# `start_configs` are the configuration to start HAL from, here identical to `al`
al = read("./init.xyz", ":")
start_configs = al

###################################
# info/arrays keys storing the DFT info of the initial database, `Fmax` can be used to exclude large forces to the design matrix
data_keys = { "E" : "energy", "F" : "forces", "V" : "virial", "Fmax" : 20.0 }

###################################
# Setting up CASTEP (DFT) calculator, any ASE calculator can be used here
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

###################################
# ACE1.jl parameters to set up a linear ACE basis
# Polynomial envelope is default, can be changed if needed
basis_info = {
    "elements" : ["Al", "Si"],
    "cor_order" : 2,
    "poly_deg_ACE" : 7,
    "poly_deg_pair" : 7,
    "r_0" : 1.8,
    "r_in" : 0.5,
    "r_cut" : 5.5 }

B = ace_basis.full_basis(basis_info);

###################################
# Isolated atom energies
E0s = { "Al" : -105.8114973092, "Si" : -163.2225204255 }

###################################
# HAL parameter info for HAL runs
run_info = {
    "niters" : 100,
    "ncomms" : 8,
    "nsteps" : 1000,
    "tau_rel" : 0.2,
    "tau_hist" : 50,
    
    "dt" : 0.5,
    "gamma" : 500.0,
    "softmax" : False,
    "tol" : 0.2,
    "eps" : 0.2,

    "swap" : True,
    "swap_step" : 10,

    "vol" : False,
    "vol_step" : 10,

    "baro" : True,
    "P" : 10.0,
    "mu" : 1e-6, 
    
    "thermo" : True,
    "T" : 500,
    "gamma" : 20.0,
}

###################################
# weights of fitting, 15/1/1 is default
weights = { "E" : 15.0, "F" : 1.0 ,"V": 1.0 }

HAL.HAL(B, E0s, weights, run_info, al, data_keys, start_configs, BayesianRidge(fit_intercept=False), calculator=calculator)
