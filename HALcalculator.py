from ase.calculators.calculator import Calculator
import numpy as np

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

Main.eval(""" 
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
""");

from julia.Main import get_com_forces, get_com_energies

Main.eval("using ASE, JuLIP, ACE1")

ASEAtoms = Main.eval("ASEAtoms(a) = ASE.ASEAtoms(a)")
ASECalculator = Main.eval("ASECalculator(c) = ASE.ASECalculator(c)")
convert = Main.eval("julip_at(a) = JuLIP.Atoms(a)")

class HALCalculator(Calculator):
    """
    ASE-compatible Calculator that calls JuLIP.jl for forces and energy
    """
    implemented_properties = ['energy', 'forces']
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
        if 'energy' in properties:
            self.results['energy'] = np.array(get_com_energies(self.julip_calculator, julia_atoms))
        if 'forces' in properties:
            self.results['forces'] = np.array(get_com_forces(self.julip_calculator, julia_atoms))