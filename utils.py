import ase
import numpy as np
import matplotlib.pyplot as plt

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

def save_pot(fname):
    Main.eval("save_dict(\"./{}\", Dict(\"IP\" => write_dict(ACE_IP)))".format(fname))

def plot_dimer(IP, elements, E0s, R = np.linspace(0.1, 8.0, 100), m=0, save=True):
    plt.figure()
    minE = 0
    for (i,el1) in enumerate(elements):
        for (j,el2) in enumerate(elements):
            if i >= j:
                E = []
                for r in R:
                    at = ase.Atoms("{}{}".format(el1, el2), cell=np.eye(3)*100, positions=[[0.0,0.0,0.0], [0.0, 0.0, r]])
                    at.set_calculator(IP)
                    E.append(at.get_potential_energy() - E0s[el1] - E0s[el2])
                if np.min(E) < minE:
                    minE = np.min(E)
                plt.plot(R, E, label="{}-{}".format(el1, el2))

    print(minE)
    if minE < -5.0:
        minE = -5.0
    
    plt.ylim(1.2 * minE, -1.2 * minE)
    plt.ylabel("Energy [eV/atom]", fontsize=14)
    plt.xlabel("Interatomic distance [Å]", fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("dimer_{}.pdf".format(m))