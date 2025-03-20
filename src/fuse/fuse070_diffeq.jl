function model_rhs!(du, u, params::DiffEqParams{Fuse070Parameters,Fuse070State}, t)
    S = u
    ΔS = du
    S1 = S[1]
    S2 = S[2]
    p = params.state.forcing[1]
    PET = params.state.forcing[2]
    fuse = params.parameters

    s⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    saturation = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    qsx = p * (1 - (1 - saturation)^fuse.b)
    e1 = PET * min_smooth(s⁺, 1.0, fuse.m)
    q12 = fuse.ku * saturation^fuse.c
    qufof = (p - qsx) * sigmoid_activation(S1, fuse.S1max, fuse.ω)
    qb = fuse.v * S2
    ΔS[1] = p - qsx - e1 - q12 - qufof
    ΔS[2] = q12 - qb
    return
end
