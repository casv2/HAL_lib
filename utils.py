import ase
import numpy as np
import matplotlib.pyplot as plt

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

def save_pot(fname):
    Main.eval("save_dict(\"./{}\", Dict(\"IP\" => write_dict(ACE_IP)))".format(fname))

def plot_dimer(IP, elements, E0s, R = np.linspace(0.1, 8.0, 100), m=0, dimer_data=None):
    plt.figure()

    minE = 0
    maxEs = {}
    for (i,el1) in enumerate(elements):
        for (j,el2) in enumerate(elements):
            if i >= j:
                E = []
                for r in R:
                    at = ase.Atoms("{}{}".format(el1, el2), cell=np.eye(3)*100, positions=[[0.0,0.0,0.0], [0.0, 0.0, r]])
                    at.set_calculator(IP)
                    E.append(at.get_potential_energy() - E0s[el1] - E0s[el2])
                #if np.min(E) < minE:
                #    minE = np.min(E)
                plt.plot(R, E, label="{}-{}".format(el1, el2))
                maxEs["{}-{}".format(el1, el2)] = int(np.max(E))
    
    if dimer_data is not None:
        dimer_disps = [ np.linalg.norm(at.positions[0] - at.positions[1]) for at in dimer_data]
        dimer_energies = [ at.info["energy"]  - np.sum([at.get_chemical_symbols().count(EL) * E0 for EL, E0 in E0s.items()]) for at in dimer_data]
        plt.scatter(dimer_disps, dimer_energies, color="black", s=3, label="dimer data")
        minE = np.min(dimer_energies) * 1.5
    
    if np.abs(minE) > 10:
        plt.ylim(-10, 10)
    else:
        plt.ylim(1.5 * minE, - 1.5 * minE)
    plt.title("maxE [eV]: " + str(maxEs), fontsize=8)
    plt.ylabel("Energy [eV/atom]", fontsize=10)
    plt.xlabel("Interatomic distance [Å]", fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dimer_{}.pdf".format(m))