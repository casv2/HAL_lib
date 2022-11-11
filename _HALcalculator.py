from ase.calculators.calculator import Calculator
import numpy as np

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

Main.eval(""" 
using LinearAlgebra

function get_com_forces(IPs, at)
    GC.gc()

    nIPs = length(IPs)
    Fs = Vector(undef, nIPs)
    Fs[1] = forces(IPs[1], at)

    @Threads.threads for i in 2:nIPs
        Fs[i] = forces(IPs[i], at)
    end

    return Fs
end

function get_com_energies(IPs, at)
    GC.gc()
    
    nIPs = length(IPs)
    Es = Vector(undef, nIPs)
    Es[1] = energy(IPs[1], at)

    @Threads.threads for i in 2:nIPs
        Es[i] = energy(IPs[i], at)
    end

    return Es
end

function get_bias_forces(IPs, at)
    GC.gc()

    nIPs = length(IPs)
    Es = get_com_energies(IPs, at)
    Fs = get_com_forces(IPs, at)

    varE = 0

    @Threads.threads for i in 2:nIPs
        varE += (Es[i] - Es[1])^2
    end

    varE = varE/(nIPs-1)

    Fbias = Vector(undef,nIPs-1)

    @Threads.threads for i in 2:nIPs
        Fbias[i-1] = 2*(Es[i] - Es[1])*(Fs[i] - Fs[1])
    end

    return 1/sqrt(varE) * sum(Fbias)/(nIPs-1)
end

softmax(x) = exp.(x) ./ sum(exp.(x))

function get_uncertainty(IPs, at; Freg=0.2)
    GC.gc()

    nIPs = length(IPs)
    Fs = get_com_forces(IPs, at)

    dFn = Vector(undef, nIPs-1)

    @Threads.threads for i in 2:nIPs
        dFn[i-1] = norm.(Fs[i] - Fs[1])
    end

    dFn = sum(hcat(dFn...), dims=2)/(nIPs-1)
    Fn = norm.(Fs[1])
    
    p = softmax(dFn ./ (Fn .+ Freg))

    return maximum(p)
end
""");

from julia.Main import get_uncertainty, get_com_energies, get_bias_forces

Main.eval("using ASE, JuLIP, ACE1")

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

class HALCalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['uncertainty', 'bias_forces', 'com_energies']
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
        if 'uncertainty' in properties:
            self.results['uncertainty'] = np.array(get_uncertainty(self.julip_calculator, julia_atoms))
        if 'com_energies' in properties:
            self.results['com_energies'] = np.array(get_com_energies(self.julip_calculator, julia_atoms))
        if 'bias_forces' in properties:
            self.results['bias_forces'] = np.array(get_bias_forces(self.julip_calculator, julia_atoms))