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

    all_distances = utils.distances_dict(atoms_list)
    transform_dict = {}

    r_0s, r_mins = [], []

    for (i,el1) in enumerate(elements):
        for (j,el2) in enumerate(elements):
            if i >= j:
                try:
                    d = all_distances[(el1, el2)]
                except:
                     d = all_distances[(el2, el1)]
                
                r_min = np.min(d)
                x,y = np.histogram(d, bins=50)

                try:
                    r_0 = y[argrelextrema(x, np.greater)[0][0]]
                except:
                    r_0 = y[np.argmax(x)]

                transform_dict[(el1, el2)] = {}
                transform_dict[(el1, el2)]["r_min"] = r_min
                transform_dict[(el1, el2)]["r_0"] = r_0
                r_0s.append(r_0)
                r_mins.append(r_min)

    r_0_av = np.mean(r_0s)
    r_in_min = np.min(r_mins)

    r_cut = [4.5, 8.5]

    print("transform dict: ", transform_dict)

    print("r_in {}, r_0 : {}".format(r_in_min, r_0_av))

    @timeout_decorator.timeout(optim_basis_param["timeout"], use_signals=True)   

    def objective(trial,
                        r_cut = r_cut,
                        r_0_av=r_0_av, 
                        r_in_min=r_in_min,
                        max_len_B=max_len_B,
                        transform_dict=transform_dict, 
                        max_deg_D=max_deg_D):

        cor_order = trial.suggest_int('cor_order', low=2, high=4)

        if cor_order in max_deg_D:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=max_deg_D[cor_order])
        else:
            maxdeg = trial.suggest_int('maxdeg', low=3, high=20)

        r_cut = trial.suggest_float('r_cut', low=r_cut[0], high=r_cut[1])

        basis_info = {
        "elements" : elements, 
        "cor_order" : cor_order,          
        "maxdeg" : maxdeg,
        "transform_dict" : transform_dict,
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
            k, n = len(c), len(Y)
            return (n * np.log(np.sum(np.power(Psi @ c - Y, 2))/n)) + (k * np.log(n))
        elif optim_basis_param["IC"] == "AIC":
            k, n = len(c), len(Y)
            return (n * np.log(np.sum(np.power(Psi @ c - Y, 2))/n)) + 2*k 
        elif optim_basis_param["IC"] == "AICc":
            k, n = len(c), len(Y)
            return (n * np.log(np.sum(np.power(Psi @ c - Y, 2))/n)) + 2*k + (2 * (k**2) + (2 * k)) / (n - k - 1)
            
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
