from ase.calculators.calculator import Calculator
import numpy as np

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

Main.eval(""" 
using LinearAlgebra

function get_com_energies(CO_IP, at)
    GC.gc()
    
    E_bar, E_comms = ACE1.co_energy(CO_IP, at)
    return E_bar, E_comms
end

function get_all_data(CO_IP, at)
    GC.gc()

    nats = length(at)
    E_bar, E_comms = ACE1.co_energy(CO_IP, at)
    F_bar, F_comms = ACE1.co_forces(CO_IP, at)

    nIPs = length(E_comms)

    varE = 0
    @Threads.threads for i in 1:nIPs
        varE += (E_comms[i] - E_bar)^2
    end

    varE = varE/(nIPs)

    Fbias = [ zeros(SVec{3,Float64}) for i in 1:nats` ]

    @Threads.threads for j in 1:nats
        @Threads.threads for i in 1:nIPs
            Fbias[j] += 2*(E_comms[i] - E_bar)*(F_comms[i][j] - F1[j])
        end
    end

    dFn = zeros(nats)

    @Threads.threads for j in 1:nats
        @Threads.threads for i in 1:nIPs
            dFn[j] += norm(F_comms[i][j] - F_bar[j])
        end
    end

    return F_bar, 1/sqrt(varE) * (Fbias/(nIPs)), dFn/(nIPs)
end
""");

from julia.Main import get_all_data, get_com_energies

Main.eval("using ASE, JuLIP, ACE1")

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

class COcalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['all_data', 'com_energies']
    default_parameters = {}
    name = 'JulipCalculator'

    def __init__(self, julip_calculator):
        Calculator.__init__(self)
        self.julip_calculator = Main.eval(julip_calculator) #julia.eval

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        julia_atoms = ASEAtoms(atoms)
        julia_atoms = convert(julia_atoms)
        self.results = {}
        if 'all_data' in properties:
            self.results['all_data'] = get_all_data(self.julip_calculator, julia_atoms)
        if 'com_energies' in properties:
            self.results['com_energies'] = np.array(get_com_energies(self.julip_calculator, julia_atoms))
        # if 'bias_forces' in properties:
        #     self.results['bias_forces'] = np.array(get_bias_forces(self.julip_calculator, julia_atoms))