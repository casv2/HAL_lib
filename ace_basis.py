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
    Main.poly_deg_ACE = basis_info["poly_deg_ACE"]
    Main.poly_deg_pair = basis_info["poly_deg_pair"]
    Main.r_0 = basis_info["r_0"]
    Main.r_in = basis_info["r_in"]
    Main.r_cut_ACE = basis_info["r_cut_ACE"]
    Main.r_cut_pair = basis_info["r_cut_pair"]
    #Main.p_trans = basis_info["p_trans"]
    #Main.p_env = basis_info["p_env"]
    
    #Main.Dd_deg = basis_info["Dd_deg"]
    #Main.Dn_w = basis_info["Dn_w"]
    #Main.Dl_w = basis_info["Dl_w"]


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

            # Dd = Dict( "default" => Dd_deg ) #10
            # Dn = Dict( "default" => Dn_w ) #1.0
            # Dl = Dict( "default" => Dl_w ) #1.0
            # Deg = ACE1.RPI.SparsePSHDegreeM(Dn, Dl, Dd)

            # Bsite = rpi_basis(species = Symbol.(elements),
            #                     trans = PolyTransform(r_0, p_trans),
            #                     N = cor_order,
            #                     r0 = r_0,
            #                     D = Deg,
            #                     rin = r_in, 
            #                     rcut = r_cut_ACE,  
            #                     maxdeg = 1.0,
            #                     pin = 2) 
                
            # trans_r = AgnesiTransform(; r0=r_0, p = p_trans)
            # envelope_r = ACE1.PolyEnvelope(p_env, r_0, r_cut_pair)
            # Jnew = transformed_jacobi_env(poly_deg_pair, trans_r, envelope_r, r_cut_pair)
            # Bpair = PolyPairBasis(Jnew, Symbol.(elements))
        
            Bsite = rpi_basis(species = Symbol.(elements),
                                N = cor_order,       # correlation order = body-order - 1
                                maxdeg = poly_deg_ACE,  # polynomial degree
                                r0 = r_0,     # estimate for NN distance
                                rin = r_in,
                                rcut = r_cut_ACE,   # domain for radial basis (cf documentation)
                                pin = 2)      

            Bpair = pair_basis(species = Symbol.(elements),
                   r0 = r_0,
                   maxdeg = poly_deg_pair,
                   rcut = r_cut_pair,
                   rin = 0.0,
                   pin = 0 )

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
    
