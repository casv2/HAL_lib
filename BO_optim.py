import optuna 
from optuna.samplers import TPESampler
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import numpy as np
import matplotlib.pyplot as plt

from HAL_lib import lsq
from HAL_lib import ace_basis

def BO_basis_optim(optim_basis_param, solver, atoms_list, E0s, data_keys, weights, dimer_data=None, D_prior=None):#, D_max_B=None):
    elements = optim_basis_param["elements"]
    max_deg_D = optim_basis_param["max_deg_D"]
    max_len_B = optim_basis_param["max_len_B"]

    distances_all = np.hstack([ at.get_all_distances(mic=True).flatten() for at in atoms_list])
    distances_first_shell = distances_all[ distances_all <= 3.5]
    distances_non_zero = distances_first_shell[distances_first_shell != 0.0] 
    
    if "r_in" in optim_basis_param:
        r_in = optim_basis_param["r_in"]
    else:
        r_in = np.min(distances_non_zero)

    if "r_0" in optim_basis_param:
        r_0 = optim_basis_param["r_0"]
    else:
        x,y = np.histogram(distances_non_zero, bins=100)
        r_0 = y[np.argmax(x)]

    if "r_cut_ACE" in optim_basis_param:
        r_cut_ACE = optim_basis_param["r_cut_ACE"]
    else:
        r_cut_ACE = [1.5*r_0, 2.5*r_0]
    
    if "r_cut_pair" in optim_basis_param:
        r_cut_pair = optim_basis_param["r_cut_pair"]
    else:
        r_cut_pair = [2.0*r_0, 3.0*r_0]

    print("r_in {}, r_0 : {}".format(r_in, r_0))
    print("r_cut_ACE {}".format(r_cut_ACE))
    print("r_cut_pair {}".format(r_cut_pair))

    @timeout_decorator.timeout(optim_basis_param["timeout"], use_signals=True)   

    def objective(trial, r_cut_ACE=r_cut_ACE, r_cut_pair=r_cut_pair, max_len_B=max_len_B, max_deg_D=max_deg_D):

        cor_order = trial.suggest_int('cor_order', low=2, high=4)

        if cor_order in max_deg_D:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=max_deg_D[cor_order])
        else:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=22)

        r_cut_ACE = trial.suggest_float('r_cut_ACE', low=r_cut_ACE[0], high=r_cut_ACE[1])
        r_cut_pair = trial.suggest_float('r_cut_pair', low=r_cut_pair[0], high=r_cut_pair[1])

        poly_deg_pair = trial.suggest_int('poly_deg_pair', low=3, high=22)

        basis_info = {
        "elements" : elements, 
        "cor_order" : cor_order,          
        "poly_deg_pair" : poly_deg_pair,
        "r_cut_pair" : r_cut_pair,
        "maxdeg" : maxdeg,
        "r_0" : r_0,
        "r_in" : r_in,
        "r_cut_ACE" : r_cut_ACE}

        if dimer_data is None:
            B_ace, len_B_ace = ace_basis.full_basis(basis_info, return_length=True)
            Psi, Y = lsq.add_lsq(B_ace, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))

            solver.fit(Psi, Y)
            c = solver.coef_
        else:
            B_pair, len_B_pair = ace_basis.pair_basis(basis_info, return_length=True)

            Psi_pair, Y_pair = lsq.add_lsq(B_pair, E0s, dimer_data, data_keys, weights, data_keys.get('Fmax'))

            solver.fit(Psi_pair, Y_pair)
            c_prior = solver.coef_
        
            B_ace, len_B_ace = ace_basis.full_basis(basis_info, return_length=True)
            Psi, Y = lsq.add_lsq(B_ace, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))
            
            c_prior_full = np.pad(c_prior, (0, (len_B_ace - len_B_pair)))
            Y_sub = Y - (Psi @ c_prior_full)

            solver.fit(Psi, Y_sub)
            c = solver.coef_ + c_prior_full

        if len_B_ace > max_len_B:
            return -1e32
        else:
            #k = ((2 * len(c)) + 2)
            #return np.log(np.sum(np.power(Psi @ c - Y, 2)) / len(Y)) + (k/len(Y)) * np.log(len(Y))
            return len(Y)*np.log(np.sum(np.power(Psi @ c - Y, 2))) + 2*len(c)
            #return np.log(len(Y)) * len_B - 2*score

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    if D_prior is not None:
        study.enqueue_trial(D_prior)
    
    study.optimize(objective, callbacks=[MaxTrialsCallback(optim_basis_param["n_trials"], states=(TrialState.COMPLETE,))], catch=(TimeoutError,))#, show_progress_bar=True)

    D = study.best_params
    D["r_in"] = r_in
    D["r_0"] =  r_0
    D["elements"] = elements

    print("BEST BASIS, DB size: {}, VALUE: {}, STUDYSIZE: {}".format(len(atoms_list), study.best_value, len(study.trials)))

    return D
