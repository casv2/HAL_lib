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
    
    Main.r_0_av = basis_info["r_0_av"]
    Main.r_in_min = basis_info["r_in_min"]

    Main.r_cut = basis_info["r_cut"]

    if "transform_dict" in basis_info:
        Main.transform_dict = basis_info["transform_dict"]
    else:
        Main.transform_dict = {}

    Main.eval("""
            using ACE1: transformed_jacobi, transformed_jacobi_env
            using ACE1.Transforms: multitransform, transform, transform_d

            pin = 2
            pcut = 2

            ninc = (pcut + pin) * (cor_order-1)
            maxn = maxdeg + ninc 

            transforms = Dict()
            cutoffs = Dict()

            if length(keys(transform_dict)) == 0
                trans = PolyTransform(2, r_0_av)
                ninc = (pcut + pin) * (cor_order-1)
                maxn = maxdeg + ninc 
                Pr = transformed_jacobi(maxn, trans, r_cut, r_in_min; pcut = pcut, pin = pin)
            else
                for d in keys(transform_dict)
                    transforms[Symbol.(d)] = PolyTransform(2, transform_dict[d]["r_0"])
                    cutoffs[Symbol.(d)] = (transform_dict[d]["r_min"], r_cut) 
                end
                ace_transform = multitransform(transforms, cutoffs=cutoffs)
                Pr = transformed_jacobi(maxn, ace_transform; pcut = pcut, pin= pin )
            end
        
            D = ACE1.RPI.SparsePSHDegree()

            rpibasis = ACE1x.Pure2b.pure2b_basis(species =  AtomicNumber.(Symbol.(elements)),
                                       Rn=Pr, 
                                       D=D, 
                                       maxdeg=maxdeg, 
                                       order=cor_order, 
                                       delete2b = true)

            trans_r = AgnesiTransform(; r0=r_0_av, p = 2)

            envelope_r = ACE1.PolyEnvelope(2, r_in_min, r_cut)

            Jnew = transformed_jacobi_env(maxdeg, trans_r, envelope_r, r_cut)
            pair = PolyPairBasis(Jnew, Symbol.(elements))

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
    
