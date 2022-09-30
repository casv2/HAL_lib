from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
Main.eval("using ASE, JuLIP, ACE1")

def save_pot(fname):
    Main.eval("save_dict(\"./{}\", Dict(\"IP\" => write_dict(IP)))".format(fname))