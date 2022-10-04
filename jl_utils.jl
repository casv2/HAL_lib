module jl_utils

using ACE1, ASE, JuLIP

function get_bias_forces(IP, IPs, at)
    nIPs = length(IPs)

    E = energy(IP, at)
    F = forces(IP, at)

    Fs = Vector(undef, nIPs)
    Es = Vector(undef, nIPs)

    @Threads.threads for i in 1:nIPs
        Es[i] = energy(IPs[i], at) 
        Fs[i] = forces(IPs[i], at)
    end

    varE = sum([ (Es[i] - E)^2 for i in 1:nIPs])/nIPs

    biasF =  1/sqrt(varE) * sum([ 2*(Es[i] - E)*(Fs[i] - F) for i in 1:nIPs])/nIPs
    
    return biasF
end

end