from ase.calculators.calculator import Calculator
import numpy as np

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

Main.eval(""" 
using LinearAlgebra

function get_com_forces(CO_IP, at)
    GC.gc()
    
    F_bar, F_comms = ACE1.co_forces(CO_IP, at)

    return F_bar, F_comms
end

function get_com_energies(CO_IP, at)
    GC.gc()
    
    E_bar, E_comms = ACE1.co_energy(CO_IP, at)

    return E_bar, E_comms
end

function get_bias_forces(CO_IP, at)
    GC.gc()

    E_bar, E_comms = get_com_energies(CO_IP, at)
    F_bar, F_comms = get_com_forces(CO_IP, at)

    nIPs = length(E_comms)

    varE = 0
    @Threads.threads for i in 1:nIPs
        varE += (E_comms[i] - E_bar)^2
    end

    varE = varE/(nIPs)

    Fbias = Vector(undef,nIPs)

    @Threads.threads for i in 1:nIPs
        Fbias[i] = 2*(E_comms[i] - E_bar)*(F_comms[i] - F_bar)
    end

    return 1/sqrt(varE) * sum(Fbias)/(nIPs)
end

#softmax(x) = exp.(x) ./ sum(exp.(x))

function get_force_diff(CO_IP, at; Freg=0.2)
    GC.gc()

    F_bar, F_comms = get_com_forces(CO_IP, at)
    nIPs = length(F_comms)

    dFn = Vector(undef, nIPs)

    @Threads.threads for i in 1:nIPs
        dFn[i] = norm.(F_comms[i] - F_bar)
    end

    dFn = sum(hcat(dFn...), dims=2)/nIPs

    return dFn
end
""");

from julia.Main import get_force_diff, get_com_energies, get_bias_forces

Main.eval("using ASE, JuLIP, ACE1")

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

class COcalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['force_diff', 'bias_forces', 'com_energies']
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
        if 'force_diff' in properties:
            self.results['force_diff'] = np.array(get_force_diff(self.julip_calculator, julia_atoms))
        if 'com_energies' in properties:
            self.results['com_energies'] = np.array(get_com_energies(self.julip_calculator, julia_atoms))
        if 'bias_forces' in properties:
            self.results['bias_forces'] = np.array(get_bias_forces(self.julip_calculator, julia_atoms))