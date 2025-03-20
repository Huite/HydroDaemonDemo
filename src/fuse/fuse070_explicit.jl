function explicit_timestep!(state::Fuse070State, fuse::Fuse070Parameters, Δt)
    S = state.S
    S1 = S[1]
    S2 = S[2]
    p = state.forcing[1]
    PET = state.forcing[2]

    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    sf = clamp(S1 / fuse.S1max, 0.0, 1.0)

    qsx = p * (1 - (1 - sf)^fuse.b)
    e1 = PET * min(S⁺, 1.0)
    q12 = fuse.ku * sf^fuse.c
    qufof = (p - qsx) * activation(S1, fuse.S1max)
    qb = fuse.v * S2

    ΔS1 = p - qsx - e1 - q12 - qufof
    ΔS2 = q12 - qb
    S[1] = max(S1 + ΔS1 * Δt, 0)
    S[2] = max(S2 + ΔS2 * Δt, 0)
    return
end
