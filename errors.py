import numpy as np

def print_errors(IP, al):
    E_DFT = [at.info["energy"]/len(at) for at in al]
    F_DFT = [at.arrays["forces"] for at in al]

    E_ACE = []
    F_ACE = []

    for at in al:
        at.set_calculator(IP)
        E_ACE.append(at.get_potential_energy()/len(at))
        F_ACE.append(at.get_forces())

    E_RMSE = np.sqrt(np.mean(np.power(np.array(E_DFT) - np.array(E_ACE), 2)))
    F_RMSE = np.sqrt(np.mean(np.power(np.array(F_DFT) - np.array(F_ACE), 2)))

    print("============================================")
    print("|  E   : {} meV/at |  F  :  {}  eV/A |".format(str(E_RMSE*1E3)[:5], str(F_RMSE)[:5]))
    print("============================================")
    