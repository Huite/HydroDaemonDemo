function waterbalance!(state::Fuse070State, fuse::Fuse070Parameters)
    p = state.forcing[1]
    PET = state.forcing[2]
    S = state.S

    S1 = S[1]
    S2 = S[2]
    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    sf = clamp(S1 / fuse.S1max, 0.0, 1.0)

    qsx = p * (1 - (1 - sf)^fuse.b)
    e1 = PET * min(S⁺, 1.0)
    q12 = fuse.ku * sf^fuse.c
    qufof = (p - qsx) * activation(S1, fuse.S1max)
    qb = fuse.v * S2

    state.dS[1] = p - qsx - e1 - q12 - qufof
    state.dS[2] = q12 - qb
    return
end


function explicit_timestep!(state::Fuse070State, fuse::Fuse070Parameters, Δt)
    waterbalance!(state, fuse)
    @. state.S += state.dS * Δt
    @. state.S = max(state.S, 0)
    return
end

function residual!(
    linearsolver::LinearSolver,
    state::Fuse070State,
    fuse::Fuse070Parameters,
    Δt,
)
    waterbalance!(state, fuse)
    # Newton-Raphson use the negative residual
    @. linearsolver.rhs = -(state.dS - (state.S - state.Sold) / Δt)
    return
end

function jacobian!(linearsolver, state::Fuse070State, fuse::Fuse070Parameters, Δt)
    p = state.forcing[1]
    PET = state.forcing[2]
    S1 = state.S[1]
    J = linearsolver.J

    # Compute the terms and their derivatives.
    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    dS⁺ = 1.0 / (fuse.ϕtens * fuse.S1max)
    sf = clamp(S1 / fuse.S1max, 0.0, 1.0)
    dsf = dclamp(S1 / fuse.S1max, 0.0, 1.0)
    act = activation(S1, fuse.S1max)
    dact = dactivation(S1, fuse.S1max)

    # Apply chain rule and product rule as needed.
    de1 = PET * min(S⁺, 1.0) * dS⁺
    dq12 = fuse.c * fuse.ku * sf^(fuse.c - 1) * dsf
    qsx = p * (1 - (1 - sf)^fuse.b)
    dqsx = p * fuse.b * (1 - sf)^(fuse.b - 1) * dsf * 1.0 / fuse.S1max
    dqufof = -dqsx * act + (p - qsx) * dact
    dqb = -fuse.v

    # Fill terms in the Jacobian
    J[1, 1] = -de1 - dq12 - dqsx - dqufof - 1.0 / Δt
    J[1, 2] = 0.0
    J[2, 1] = dq12
    J[2, 2] = dqb - 1.0 / Δt
    return
end
