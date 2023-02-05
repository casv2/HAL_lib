import optuna 
from optuna.samplers import TPESampler
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from HAL_lib import lsq
from HAL_lib import ace_basis
from HAL_lib import utils

def BO_basis_optim(optim_basis_param, solver, atoms_list, E0s, data_keys, weights, D_prior=None):#, D_max_B=None):
    elements = optim_basis_param["elements"]
    max_deg_D = optim_basis_param["max_deg_D"]
    max_len_B = optim_basis_param["max_len_B"]

    distances_all = np.hstack([ at.get_all_distances(mic=True).flatten() for at in atoms_list])
    distances_first_shell = distances_all[ distances_all <= 3.5]
    distances_non_zero = distances_first_shell[distances_first_shell != 0.0] 
    
    r_in_min = np.min(distances_non_zero)

    x,y = np.histogram(distances_non_zero, bins=100)
    r_0_av = y[np.argmax(x)]

    r_cut = [4.5, 8.0]

    print("r_in {}, r_0 : {}".format(r_in_min, r_0_av))

    @timeout_decorator.timeout(optim_basis_param["timeout"], use_signals=True)   

    def objective(trial,
                        r_cut = r_cut,
                        r_0_av=r_0_av, 
                        r_in_min=r_in_min,
                        max_len_B=max_len_B, 
                        max_deg_D=max_deg_D):

        cor_order = trial.suggest_int('cor_order', low=2, high=4)

        if cor_order in max_deg_D:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=max_deg_D[cor_order])
        else:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=10)

        r_cut = trial.suggest_float('r_cut', low=r_cut[0], high=r_cut[1])

        basis_info = {
        "elements" : elements, 
        "cor_order" : cor_order,          
        "maxdeg" : maxdeg,
        "r_0_av" : r_0_av,
        "r_in_min" : r_in_min,
        "r_cut" : r_cut }

        B, len_B = ace_basis.full_basis(basis_info, return_length=True)

        Psi, Y = lsq.add_lsq(B, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))

        solver.fit(Psi, Y)
        c = solver.coef_
        #score = solver.scores_[-1]

        if len_B > max_len_B:
            raise Exception("basis too large!")
        elif optim_basis_param["IC"] == "BIC":
            k = ((2 * len(c)) + 2)
            return np.log(np.sum(np.power(Psi @ c - Y, 2)) / len(Y)) + (k/len(Y)) * np.log(len(Y))
        elif optim_basis_param["IC"] == "AIC":
            k, n= len(c), len(Y)
            return 2*k + (n * np.log(np.sum(np.power(Psi @ c - Y, 2))))
        elif optim_basis_param["IC"] == "AICc":
            k, n= len(c), len(Y)
            return 2*k + (n * np.log(np.sum(np.power(Psi @ c - Y, 2)))) + (2 * (k**2) + (2 * k)) / (n - k - 1)
            
            #return np.log(len(Y)) * len_B - 2*score

    study = optuna.create_study(sampler=TPESampler(), direction='minimize')
    if D_prior is not None:
        study.enqueue_trial(D_prior)
    
    study.optimize(objective, callbacks=[MaxTrialsCallback(optim_basis_param["n_trials"], states=(TrialState.COMPLETE,))], catch=(TimeoutError,))

    D = study.best_params
    D["r_in_min"] = r_in_min
    D["r_0_av"] =  r_0_av
    D["elements"] = elements

    print("BEST BASIS, DB size: {}, VALUE: {}, STUDYSIZE: {}".format(len(atoms_list), study.best_value, len(study.trials)))
    print(D)

    return D
