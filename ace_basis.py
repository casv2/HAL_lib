#load Julia and Python dependencies
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

def ref_pot(E0s):
    Main.E0s = E0s
    ref_pot = Main.eval("Dict(zip(Symbol.(keys(E0s)), values(E0s)))")
    return ref_pot

def full_basis(basis_info):
    Main.elements = basis_info["elements"]
    Main.cor_order = basis_info["cor_order"]
    Main.poly_deg = basis_info["poly_deg"]
    Main.r_0 = basis_info["r0"]
    Main.r_in = basis_info["r_in"]
    Main.r_cut = basis_info["r_cut"]

    B = Main.eval("""
            Bsite = rpi_basis(species = Symbol.(elements),
                                N = cor_order,       # correlation order = body-order - 1
                                maxdeg = poly_deg,  # polynomial degree
                                r0 = r_0,     # estimate for NN distance
                                rin = r_in,
                                rcut = r_cut,   # domain for radial basis (cf documentation)
                                pin = 2)                     # require smooth inner cutoff

            Bpair = pair_basis(species = Symbol.(elements),
                                r0 = r_0,
                                maxdeg = poly_deg,
                                rcut = r_cut + 1.0,
                                rin = 0.0,
                                pin = 0 )   # pin = 0 means no inner cutoff

            B = JuLIP.MLIPs.IPSuperBasis([Bpair, Bsite]);
            """)
    return B

def combine(ref_pot, ace_basis, c, comms):
    Main.ref_pot = ref_pot
    Main.B = ace_basis
    Main.c = c
    Main.comms = comms
    Main.ncomms = len(comms)
    print(len(comms))
    print(len(comms[1]))
    print(len(c))
    print(comms)

    IP = Main.eval("IP = JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, c))")
    IPs = Main.eval("IPs = [JuLIP.MLIPs.SumIP(ref_pot, JuLIP.MLIPs.combine(B, comms[i, :])) for i in 1:ncomms]")
    return IP, IPs
    
