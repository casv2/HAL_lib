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

def BO_basis_optim(optim_basis_param, solver, atoms_list, E0s, data_keys, weights, D_prior=None):#, D_max_B=None):
    elements = optim_basis_param["elements"]
    max_deg_D = optim_basis_param["max_deg_D"]

    distances_all = np.hstack([ at.get_all_distances(mic=True).flatten() for at in atoms_list])
    distances_first_shell = distances_all[ distances_all <= 3.5]
    distances_non_zero = distances_first_shell[distances_first_shell != 0.0] 

    r_in = np.min(distances_non_zero)

    x,y = np.histogram(distances_non_zero, bins=100)
    r_0 = y[np.argmax(x)]

    print("r_in {}, r_0 : {}".format(r_in, r_0))

    @timeout_decorator.timeout(optim_basis_param["timeout"], use_signals=True)   

    def objective(trial, max_deg_D=max_deg_D):

        cor_order = trial.suggest_int('cor_order', low=2, high=4)

        if cor_order in max_deg_D:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=max_deg_D[cor_order])
        else:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=22)

        r_cut_ACE = trial.suggest_float('r_cut_ACE', low=1.5*r_0, high=2.5*r_0)
        r_cut_pair = trial.suggest_float('r_cut_pair', low=2*r_0, high=3.0*r_0)

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

        B, len_B = ace_basis.full_basis(basis_info, return_length=True) 
        print(len_B)

        Psi, Y = lsq.add_lsq(B, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))

        solver.fit(Psi, Y)
        score = solver.scores_[-1]

        return score

    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
    if D_prior is not None:
        study.enqueue_trial(D_prior)
    
    study.optimize(objective, callbacks=[MaxTrialsCallback(optim_basis_param["n_trials"], states=(TrialState.COMPLETE,))], catch=(TimeoutError,))#, show_progress_bar=True)

    D = study.best_params
    D["r_in"] = r_in
    D["r_0"] =  r_0
    D["elements"] = elements

    print("BEST BASIS, DB size: {}, VALUE: {}, STUDYSIZE: {}".format(len(atoms_list), study.best_value, len(study.trials)))

    return D
