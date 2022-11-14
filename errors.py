import numpy as np

def print_errors(IP, al, data_keys, CO_IP, eps):
    E_DFT = [at.info[data_keys["E"]]/len(at) for at in al]
    F_DFT = []
    for at in al:
        F_DFT.extend(at.arrays[data_keys["F"]])

    E_ACE = []
    F_ACE = []

    rel_F_err = []

    for at in al:
        at.set_calculator(IP)
        E_ACE.append(at.get_potential_energy()/len(at))
        F_ACE.extend(at.get_forces())

        F_bar, F_bias, F_bar_norms, F_bias_norms, dFn  = CO_IP.get_property('force_data',  at)
        p = dFn / (F_bar_norms + eps)
        rel_F_err.extend(p)

    E_RMSE = np.sqrt(np.mean(np.power(np.asarray(E_DFT) - np.asarray(E_ACE), 2)))
    F_RMSE = np.sqrt(np.mean(np.power(np.asarray(F_DFT) - np.asarray(F_ACE), 2)))
    rel_F_err_max = np.max(np.asarray(rel_F_err))

    l = f"|  E   : {E_RMSE*1.0e3:7.3f} meV/at |  F  :  {F_RMSE:7.3f}  eV/A |  rel_F_max : {rel_F_err_max:7.3f} |"
    print("=" * len(l))
    print(l)
    print("=" * len(l))
