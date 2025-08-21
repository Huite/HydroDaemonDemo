function waterbalance!(dS, S, fuse::Fuse070Parameters)
    p = fuse.currentforcing[1]
    PET = fuse.currentforcing[2]

    S1 = S[1]
    S2 = S[2]

    # Compuate saturation variables
    # saturation of tension storage may be > 1.0 ("super-saturation")
    S1_sat_tens = S1 / (fuse.ϕtens * fuse.S1max)
    # Ensure saturation is [0.0 - 1.0], since exponentiation of a negative
    # number results in a domain error (requires complex numbers).
    S1_sat = clamp_smooth(S1, 0.0, fuse.S1max, fuse.ω) / fuse.S1max
    act = sigmoid_activation(S1, fuse.S1max, fuse.ω)

    # Runoff
    qsx = p * (1 - (1 - S1_sat)^fuse.b)

    # Evaporation
    e1 = PET * min_smooth(S1_sat_tens, 1.0, FUSE_RHO)

    # Drainage
    q12 = fuse.ku * S1_sat^fuse.c

    # Baseflow
    qb = fuse.v * S2

    # Overflow
    qufof = (p - qsx) * act

    dS[1] = p - qsx - e1 - q12 - qufof
    dS[2] = q12 - qb
    return q12, qsx + qufof + qb
end


function dwaterbalance!(J, S, fuse::Fuse070Parameters)
    p = fuse.currentforcing[1]
    PET = fuse.currentforcing[2]
    S1 = S[1]

    # Compute the terms and their derivatives.
    S1_sat_tens = S1 / (fuse.ϕtens * fuse.S1max)
    dS1_sat_tens = 1.0 / (fuse.ϕtens * fuse.S1max)
    S1_sat = clamp_smooth(S1, 0.0, fuse.S1max, fuse.ω) / fuse.S1max
    dS1_sat = dclamp_smooth(S1, 0.0, fuse.S1max, fuse.ω) / fuse.S1max
    act = sigmoid_activation(S1, fuse.S1max, fuse.ω)
    dact = dsigmoid_activation(S1, fuse.S1max, fuse.ω)

    # Apply chain rule and product rule as needed.
    # Use safepow to avoid exponentiation of 0^(negative number)
    de1 = PET * dmin_smooth(S1_sat_tens, 1.0, FUSE_RHO) * dS1_sat_tens
    dq12 = fuse.c * fuse.ku * safepow(S1_sat, fuse.c - 1) * dS1_sat
    qsx = p * (1 - (1 - S1_sat)^fuse.b)
    dqsx = p * fuse.b * safepow(1 - S1_sat, fuse.b - 1) * dS1_sat
    dqufof = -dqsx * act + (p - qsx) * dact
    dqb = fuse.v

    # Fill terms in the Jacobian
    J[1, 1] = -de1 - dq12 - dqsx - dqufof
    J[1, 2] = 0.0
    J[2, 1] = dq12
    J[2, 2] = -dqb
    return
end

function handle_excess!(state::FuseState, fuse::Fuse070Parameters)
    (; S, flows) = state
    excess1 = max(0, S[1] - fuse.S1max)
    S[1] -= excess1
    # Add total excess to outflow
    flows[2] += excess1
    return
end
