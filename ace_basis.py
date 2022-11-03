#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

from HAL_lib import ACEcalculator
from HAL_lib import HALcalculator

def full_basis(basis_info):
    Main.elements = basis_info["elements"]
    Main.cor_order = basis_info["cor_order"]
    Main.poly_deg_ACE = basis_info["poly_deg_ACE"]
    Main.poly_deg_pair = basis_info["poly_deg_pair"]
    Main.r_0 = basis_info["r0"]
    Main.r_in = basis_info["r_in"]
    Main.r_cut = basis_info["r_cut"]

    B = Main.eval("""
            using ACE1: transformed_jacobi, transformed_jacobi_env

            Bsite = rpi_basis(species = Symbol.(elements),
                                N = cor_order,       # correlation order = body-order - 1
                                maxdeg = poly_deg_ACE,  # polynomial degree
                                r0 = r_0,     # estimate for NN distance
                                rin = r_in,
                                rcut = r_cut,   # domain for radial basis (cf documentation)
                                pin = 2)                     # require smooth inner cutoff

            trans_r = AgnesiTransform(; r0=r_0, p = 2)
            envelope_r = ACE1.PolyEnvelope(2, r_0, r_cut + 1.0)
            Jnew = transformed_jacobi_env(poly_deg_pair, trans_r, envelope_r, r_cut + 1.0)

            Bpair = PolyPairBasis(Jnew, Symbol.(elements))

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
            """)
    return B

def combine(B, c, E0s, comms):
    Main.E0s = E0s
    Main.ref_pot = Main.eval("refpot = OneBody(" + "".join([" :{} => {}, ".format(key, value) for key, value in E0s.items()]) + ")")
    Main.B = B
    Main.c = c
    Main.comms = comms
    Main.ncomms = len(comms)

    IP = Main.eval("ACE_IP = JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, c))")
    IPs = Main.eval("HAL_IP = vcat(ACE_IP, [JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, comms[i, :])) for i in 1:ncomms])")
    return ACEcalculator.ACECalculator("ACE_IP"), HALcalculator.HALCalculator("HAL_IP") 
    
