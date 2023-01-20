#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

from HAL_lib import ACEcalculator
from HAL_lib import COcalculator

def full_basis(basis_info, return_length=False):
    Main.elements = basis_info["elements"]
    Main.cor_order = basis_info["cor_order"]
    #Main.poly_deg_ACE = basis_info["poly_deg_ACE"]
    Main.poly_deg_pair = basis_info["poly_deg_pair"]
    
    Main.r_01 = basis_info["r_01"]
    Main.r_02 = basis_info["r_02"]
    Main.r_03 = basis_info["r_03"]
    Main.r_04 = basis_info["r_04"]
    Main.r_05 = basis_info["r_05"]
    Main.r_06 = basis_info["r_06"]

    Main.r_0_env = basis_info["r_0_env"]
    Main.r_in = basis_info["r_in"]
    Main.r_cut_ACE = basis_info["r_cut_ACE"]
    Main.r_cut_pair = basis_info["r_cut_pair"]
    
    Main.Dn_w = basis_info["Dn_w"]
    Main.Dl_w = basis_info["Dl_w"]

    Main.Dd_deg = basis_info["Dd_deg"]
    Main.Dd_1 = basis_info["Dd_1"]
    Main.Dd_2 = basis_info["Dd_2"]
    Main.Dd_3 = basis_info["Dd_3"]
    Main.Dd_4 = basis_info["Dd_4"]


    # Main.eval("""
    #         using ACE1: transformed_jacobi, transformed_jacobi_env

    #         Bsite = rpi_basis(species = Symbol.(elements),
    #                             N = cor_order,       # correlation order = body-order - 1
    #                             maxdeg = poly_deg_ACE,  # polynomial degree
    #                             r0 = r_0,     # estimate for NN distance
    #                             rin = r_in,
    #                             rcut = r_cut_ACE,   # domain for radial basis (cf documentation)
    #                             pin = 2)                     # require smooth inner cutoff

    #         trans_r = AgnesiTransform(; r0=r_0, p = p_trans)
    #         envelope_r = ACE1.PolyEnvelope(p_env, r_0, r_cut_pair)
    #         Jnew = transformed_jacobi_env(poly_deg_pair, trans_r, envelope_r, r_cut_pair)

    #         Bpair = PolyPairBasis(Jnew, Symbol.(elements))
            
    #         B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

    #         basis_length = length(B)
    #         """)

    Main.eval("""
            using ACE1: transformed_jacobi, transformed_jacobi_env
            using ACE1.Transforms: multitransform, transform, transform_d

            Dd = Dict("default" => Dd_deg,
            1 => Dd_1,
            2 => Dd_2,
            3 => Dd_3,
            4 => Dd_4,)
      
            Dn = Dict( "default" => Dn_w ) 
            Dl = Dict( "default" => Dl_w ) 

            Deg = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)            
        
            Bsite = rpi_basis(species = Symbol.(elements),
                   N = cor_order,
                   r0 = r_0,
                   D = Deg,
                   rin = r_in, rcut = r_cut_ACE,  
                   maxdeg = 1.0,
                   pin = 2)     # require smooth inner cutoff 

            transforms = Dict(
                    (:I, :I) => AgnesiTransform(; r0=r_01, p = 2),
                    (:I, :Pb) => AgnesiTransform(; r0=r_02, p = 2),
                    (:I, :Cs) => AgnesiTransform(; r0=r_03, p = 2),
                    (:Cs, :Cs) => AgnesiTransform(; r0=r_04, p = 2),
                    (:Cs, :Pb) => AgnesiTransform(; r0=r_05, p = 2),
                    (:Pb, :Pb) => => AgnesiTransform(; r0=r_06, p = 2),
                )

            trans_r = multitransform(transforms)

            #trans_r = AgnesiTransform(; r0=r_0, p = 2)
            envelope_r = ACE1.PolyEnvelope(2, r_0_env, r_cut_pair)
            Jnew = transformed_jacobi_env(poly_deg_pair, trans_r, envelope_r, r_cut_pair)

            Bpair = PolyPairBasis(Jnew, Symbol.(elements))

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);

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
    
