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

function get_force_data(CO_IP, at)
    GC.gc()

    nats = length(at)
    E_bar, E_comms = ACE1.co_energy(CO_IP, at)
    F_bar, F_comms = ACE1.co_forces(CO_IP, at)

    nIPs = length(E_comms)

    varE = 0
    @sync for i in 1:nIPs
        varE += (E_comms[i] - E_bar)^2
    end

    varE = varE/(nIPs)

    Fbias = [ zeros(SVec{3,Float64}) for i in 1:nats ]
    dFn = zeros(nats)

    @sync for j in 1:nats, i in 1:nIPs
        Fbias[j] += 2*(E_comms[i] - E_bar)*(F_comms[i][j] - F_bar[j])
        dFn[j] += norm(F_comms[i][j] - F_bar[j])
    end

    return F_bar, 1/sqrt(varE) * (Fbias/(nIPs)), dFn/(nIPs)
end
""");

from julia.Main import get_force_data, get_com_energies

Main.eval("using ASE, JuLIP, ACE1")

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

class COcalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['force_data', 'com_energies']
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
        if 'force_data' in properties:
            self.results['force_data'] = get_force_data(self.julip_calculator, julia_atoms)
        if 'com_energies' in properties:
            self.results['com_energies'] = get_com_energies(self.julip_calculator, julia_atoms)