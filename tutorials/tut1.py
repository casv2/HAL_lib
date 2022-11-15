from sklearn.linear_model import BayesianRidge, ARDRegression
from ase.calculators.castep import Castep
from HAL_lib import HAL
from HAL_lib import ace_basis
from ase.io import read, write

###################################
# Reading in initial database (`al`), formed of 1-10 configurations 
# `start_configs` are the configuration to start HAL from, here identical to `al`, it is advised to start varying sized supercells of relaxed configs
al = read("./init.xyz", ":")
start_configs = al

# info/arrays keys storing the DFT info of the initial database, `Fmax` can be used to exclude large forces to the design matrix
data_keys = { "E" : "energy", "F" : "forces", "V" : "virial", "Fmax" : 20.0 }

###################################
# Isolated atom energies
E0s = { "Al" : -105.8114973092, "Si" : -163.2225204255 }

###################################
# weights of fitting, 15/1/1 is default
weights = { "E" : 15.0, "F" : 1.0 ,"V": 1.0 }

###################################
# ACE1.jl parameters to set up a linear ACE basis
# Polynomial envelope is default, can be changed if needed
basis_info = {
    "elements" : ["Al", "Si"],    # elements in ACE basis
    "cor_order" : 2,              # maximum correlation order 
    "poly_deg_ACE" : 12,           # polynomial degree in ACE basis
    "poly_deg_pair" : 12,          # polynomial degree in auxiliary pair potential
    "r_0" : 1.8,                  # typical nearest neighbour distance
    "r_in" : 0.5,                 # ACE inner cutoff (0.5 is default)
    "r_cut" : 5.5 }               # ACE outer cutoff (4.5-5.5 is default) (pair outer cutoff = ACE cutoff + 1.0 Å)

B = ace_basis.full_basis(basis_info);

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
# HAL parameter info for HAL runs
run_info = {
    "niters" : 5,               # number of iterations per start config in `start_configs`
    "ncomms" : 8,                 # number of committee members (8 is default)
    "nsteps" : 1000,              # max number of exploratory HAL steps until QM/DFT calculation is triggered 
    
    "tau_rel" : 0.2,              # "fractional" relative biasing strength relative to regular MD forces
    "tau_hist" : 50,              # (burn-in) history used to tune biasing strength
    
    "dt" : 0.5,                   # timestep (in fs)
    "softmax" : False,            # softmax normalisation of relative (force) uncertainty, (default is False)
    "tol" : 0.2,                  # relative force uncertainty tolerance (0.2 is default)
    "eps" : 0.2,                  # regularising constant in relative uncertainty F_var / (F_bar + eps)

    "swap" : True,                # MC atoms swaps (random alloys) (default is False)
    "swap_step" : 10,             # occurence of MC swap steps

    "vol" : False,                # MC volume steps (default is False)
    "vol_step" : 10,              # occurence of MC volume steps

    "baro" : True,                # barostat
    "P" : 10.0,                   # pressure (in GPa)
    "mu" : 1e-6,                  # barostat control parameter
    
    "thermo" : True,              # thermostat
    "T" : 500,                    # temperature (in K)
    "gamma" : 20.0,               # thermostat control parameter
}

## scikit-learn solver option, either BRR or ARD, make sure to set fit_intercept=False and compute_score=True in either (default is BRR)
solver = BayesianRidge(fit_intercept=False, compute_score=True)
#solver = ARDRegression(fit_intercept=False, compute_score=True)

HAL.HAL(B, E0s, weights, run_info, al, data_keys, start_configs, solver, calculator=calculator)
