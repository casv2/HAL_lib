import optuna 
from optuna.samplers import TPESampler
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
from optuna.trial._state import TrialState

from HAL_lib import lsq
from HAL_lib import ace_basis

def BO_basis_optim(optim_basis_param, solver, atoms_list, E0s, data_keys, weights, D_prior=None):#, D_max_B=None):
    elements = optim_basis_param["elements"]
    max_len_B = optim_basis_param["max_len_B"]

    @timeout_decorator.timeout(optim_basis_param["timeout"], use_signals=True)   

    def objective(trial, max_len_B=max_len_B):
        """return the f1-score"""
        cor_order = trial.suggest_int('cor_order', low=2, high=4, step=1)

        r_0 = trial.suggest_float('r_0', low=1.0, high=3.5, step=0.1)

        r_in = trial.suggest_float('r_in', low=0.5, high=3.5, step=0.1)
        r_cut_ACE = trial.suggest_float('r_cut_ACE', low=4.5, high=8.0, step=0.1)

        # Dn_w = trial.suggest_float('Dn_w', low=1.0, high=1.0, step=0.1)
        # Dl_w = trial.suggest_float('Dl_w', low=1.0, high=1.4, step=0.1)

        maxdeg = trial.suggest_int('maxdeg', low=4, high=16, step=1)
        # Dd_1 = trial.suggest_float('Dd_1', low=4, high=16, step=1)
        # Dd_2 = trial.suggest_float('Dd_2', low=4, high=16, step=1)
        # Dd_3 = trial.suggest_float('Dd_3', low=4, high=16, step=1)
        # Dd_4 = trial.suggest_float('Dd_4', low=4, high=16, step=1)

        poly_deg_pair = trial.suggest_int('poly_deg_pair', low=7, high=16, step=1)
        r_cut_pair = trial.suggest_float('r_cut_pair', low=6.0, high=8.0, step=0.1)

        basis_info = {
        "elements" : elements,    # elements in ACE basis
        "cor_order" : cor_order,              # maximum correlation order 
        "poly_deg_pair" : poly_deg_pair,
        "r_cut_pair" : r_cut_pair,
        "maxdeg" : maxdeg,
        "r_0" : r_0,
        "r_in" : r_in,# typical nearest neighbour distance
        "r_cut_ACE" : r_cut_ACE}

        B, len_B = ace_basis.full_basis(basis_info, return_length=True) 

        if len_B < max_len_B:
            Psi, Y = lsq.add_lsq(B, E0s, atoms_list, data_keys, weights, data_keys.get('Fmax'))

            solver.fit(Psi, Y)
            score = solver.scores_[-1]

            return score
        else:
            return -1e32

    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
    if D_prior is not None:
        study.enqueue_trial(D_prior)

    # if D_max_B is not None:
    #     study.tell(D_max_B)
    
    study.optimize(objective, n_trials=optim_basis_param["n_trials"], catch=(TimeoutError,))#, show_progress_bar=True)

    D = study.best_params
    D["elements"] = elements

    #D_max_B = [ study.trials[i] for i in range(len(study.trials)) if study.trials[i].value == -1e32 or study.trials[i].state == TrialState.FAIL ]

    print("BEST BASIS, DB size: {}, VALUE: {}, STUDYSIZE: {}".format(len(atoms_list), study.best_value, len(study.trials)))
    print(D)

    return D#, D_max_B
