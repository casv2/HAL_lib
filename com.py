import numpy as np

def get_F_bias(IP, IPs, at):
    ncomms = len(IPs)
    at.set_calculator(IP)

    E_bar = at.get_potential_energy()
    F_bar = at.get_forces()
    
    E_comms = []
    F_comms = []

    for IP_com in IPs:
        at.set_calculator(IP_com)
        E_comms.append(at.get_potential_energy())
        F_comms.append(at.get_forces())

    E_var = (1.0/ncomms)*np.sum([np.power((E_comms[i] - E_bar),2) for i in range(ncomms)])
    F_bias = 1/np.sqrt(E_var) * np.sum([2 * (E_comms[i] - E_bar) * (F_comms[i] - F_bar).T for i in range(ncomms)], axis=0)/ncomms

    return F_bar, F_bias.T

def softmax_func(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_fi(IP, IPs, at, softmax=False):
    at.set_calculator(IP)

    F_bar = at.get_forces()
    F_comms = []

    for IP_com in IPs:
        at.set_calculator(IP_com)
        F_comms.append(at.get_forces())

    fi = np.sum([np.linalg.norm(F_bar - F_comms[i], axis=1) for i in range(len(IPs))], axis=0)/len(IPs) / (np.linalg.norm(F_bar, axis=1) + np.mean(np.linalg.norm(F_bar, axis=1)))

    if softmax:
        return np.max(softmax_func(fi))
    else:
        return np.max(fi)

