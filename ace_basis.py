#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1, ACE1x")

from HAL_lib import ACEcalculator
from HAL_lib import COcalculator

def full_basis(basis_info, return_length=False):
    Main.elements = basis_info["elements"]
    Main.cor_order = basis_info["cor_order"]
    Main.maxdeg = basis_info["maxdeg"]
    Main.poly_deg_pair = basis_info["poly_deg_pair"]
    Main.r_0 = basis_info["r_0"]
    Main.r_in = basis_info["r_in"]
    Main.r_cut_ACE = basis_info["r_cut_ACE"]
    Main.r_cut_pair = basis_info["r_cut_pair"]

    Main.eval("""
            using ACE1: transformed_jacobi, transformed_jacobi_env

            pin = 2
            pcut = 2
            ninc = (pcut + pin) * (cor_order-1)
            maxn = maxdeg + ninc 

            trans = PolyTransform(1, r_0)
            Pr = transformed_jacobi(maxn, trans, r_cut_ACE, r_in; pcut = pin, pin = pin)
            
            D = ACE1.RPI.SparsePSHDegree()

            rpibasis = ACE1x.Pure2b.pure2b_basis(species = AtomicNumber.(Symbol.(elements)),
                                       Rn=Pr, 
                                       D=D,
                                       maxdeg=maxdeg, 
                                       order=cor_order, 
                                       delete2b = true)

            trans_r = AgnesiTransform(; r0=r_0, p = 2)

            pair = pair_basis(species = Symbol.(elements),
                   r0 = r_0,
                   trans=trans,
                   maxdeg = poly_deg_pair,
                   rcut = r_cut_pair,
                   rin = 0.0,
                   pin = 0 )

            # envelope_r = ACE1.PolyEnvelope(2, r_in - 0.05, r_cut_pair)
            # Jnew = transformed_jacobi_env(poly_deg_pair, trans_r, envelope_r, r_cut_pair)
            # pair = PolyPairBasis(Jnew, Symbol.(elements))

            B = JuLIP.MLIPs.IPSuperBasis([pair, rpibasis]);

            basis_length = length(B)
            """)

    if return_length == True:
        return Main.B, Main.basis_length
    else:
        return Main.B

def combine(B, c, E0s, comms):
    Main.E0s = E0s
    Main.ref_pot = Main.eval("refpot = OneBody(" + "".join([" :{} => {}, ".format(key, value) for key, value in E0s.items()]) + ")")
    Main.B = B
    Main.c = c
    Main.comms = comms
    Main.ncomms = len(comms)

    IP = Main.eval("ACE_IP = JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, c))")
    IPs = Main.eval("CO_IP = ACE1.committee_potential(B, c, transpose(comms))")
    #Main.eval("Bpair_com = ACE1.committee_potential(Bpair, c[1:length(Bpair)], transpose(comms[:,1:length(Bpair)]))")
    #Main.eval("Bsite_com = ACE1.committee_potential(Bsite, c[length(Bpair)+1:end], transpose(comms[:, length(Bpair)+1:end]))")
    #IPs = Main.eval("CO_IP = JuLIP.MLIPs.SumIP(Bpair_com, Bsite_com)")
    return ACEcalculator.ACECalculator("ACE_IP"), COcalculator.COcalculator("CO_IP") 
    
