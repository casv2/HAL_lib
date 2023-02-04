import re
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
    
    if np.abs(minE) > 5:
        plt.ylim(-5, 5)
    else:
        plt.ylim(1.2 * minE, - 1.2 * minE)
    plt.ylabel("Energy [eV/atom]", fontsize=14)
    plt.xlabel("Interatomic distance [Å]", fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig("dimer_{}.pdf".format(m))

def get_pair_dists(all_dists, at_nos, atno1, atno2):
    dists = []
    for i in range(len(all_dists[at_nos==atno1])):
        dists.append(np.array(all_dists[at_nos==atno1][i][at_nos==atno2]))

    dists = np.array(dists)
    if atno1==atno2:
        dists = np.triu(dists)
        dists = dists[dists!=0]
    else:
        dists = dists.flatten()
    return dists


def distances_dict(at_list):
    dist_hist = {}
    for at in at_list:
        all_dists = at.get_all_distances(mic=True)

        at_nos = at.get_atomic_numbers()
        at_syms = np.array(at.get_chemical_symbols())

        formula = at.get_chemical_formula()
        unique_symbs = natural_sort(list(ase.formula.Formula(formula).count().keys()))
        for idx1, sym1 in enumerate(unique_symbs):
            for idx2, sym2 in enumerate(unique_symbs[idx1:]):
                label = (sym1,sym2)

                atno1 = at_nos[at_syms == sym1][0]
                atno2 = at_nos[at_syms == sym2][0]

                if label not in dist_hist.keys():
                    dist_hist[label] = np.array([])

                distances = get_pair_dists(all_dists, at_nos, atno1, atno2)
                dist_hist[label] = np.concatenate([dist_hist[label], distances])
    return dist_hist


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)