import numpy as np

def get_F_bias(HAL_IP, at):
    at.set_calculator(HAL_IP)
    E_comms = at.get_potential_energy()
    F_comms = at.get_forces()

    ncomms = len(E_comms)-1

    E_bar = E_comms[0]
    F_bar = F_comms[0]

    E_var = (1.0/ncomms)*np.sum([np.power((E_comms[i] - E_bar),2) for i in range(1, ncomms)])
    F_bias = 1/np.sqrt(E_var) * np.sum([2 * (E_comms[i] - E_bar) * (F_comms[i] - F_bar).T for i in range(ncomms)], axis=0)/ncomms

    return F_bar, F_bias.T

def softmax_func(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_fi(HAL_IP, at, softmax=False):
    at.set_calculator(HAL_IP)
    F_comms = at.get_forces()

    ncomms = len(F_comms)

    F_bar = F_comms[0]

    for IP_com in IPs:
        at.set_calculator(IP_com)
        F_comms.append(at.get_forces())

    fi = np.sum([np.linalg.norm(F_bar - F_comms[i], axis=1) for i in range(1, len(ncomms))], axis=0)/len(ncomms) / (np.linalg.norm(F_bar, axis=1) + np.mean(np.linalg.norm(F_bar, axis=1)))

    if softmax:
        return np.max(softmax_func(fi))
    else:
        return np.max(fi)

