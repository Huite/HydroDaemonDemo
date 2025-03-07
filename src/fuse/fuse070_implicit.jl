struct Fuse070ImplicitState
    S::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function synchronize!(state::Fuse070ImplicitState, parameters)
    return
end

function copy_state!(state::Fuse070ImplicitState)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::Fuse070ImplicitState)
    copyto!(state.S, state.Sold)
    return
end

function residual!(dS, S, params, t)
    fuse = params.modelparams
    p = params.forcing[:p]
    PET = params.forcing[:PET]
    S1 = S[1]
    S2 = S[2]

    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    sf = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)

    qsx = p * (1 - (1 - sf)^fuse.b)
    e1 = PET * min_smooth(S⁺, 1.0, fuse.m)
    q12 = fuse.ku * sf^fuse.c
    qufof = (p - qsx) * sigmoid_activation(S1, fuse.S1max, fuse.ω)
    qb = fuse.v * S2

    dS[1] = p - qsx - e1 - q12 - qufof
    dS[2] = q12 - qb
    return
end

function jacobian!(linearsolver, state::Fuse070ImplicitState, fuse::Fuse070Parameters, Δt)
    p = state.forcing[1]
    PET = state.forcing[2]
    S1 = state.S[1]

    # Compute the terms and their derivatives.
    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    dS⁺ = 1.0 / (fuse.ϕtens * fuse.S1max)
    sf = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    dsf = clamp_smooth_derivative(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    act = sigmoid_activation(S1, fuse.S1max, fuse.ω)
    dact = sigmoid_activation_derivative(S1, fuse.S1max, fuse.ω)

    # Apply chain rule and product rule as needed.
    de1 = PET * min_smooth_derivative(S⁺, 1.0, fuse.m) * dS⁺
    dq12 = fuse.c * fuse.ku * sf^(fuse.c - 1) * dsf
    qsx = p * (1 - (1 - sf)^fuse.b)
    dqsx = fuse.p * fuse.b * (1 - sf)^(fuse.b - 1) * dsf * 1.0 / fuse.Smax
    dqufof = -dqsx * act + (fuse.p - qsx) * dact
    dqb = fuse.v

    # Fill terms in the Jacobian
    J = linearsolver.J
    J[1, 1] = -de1 - dq12 - dqsx - dqufof
    J[1, 2] = 0.0
    J[2, 1] = dq12
    J[2, 2] = dqb
    return
end
