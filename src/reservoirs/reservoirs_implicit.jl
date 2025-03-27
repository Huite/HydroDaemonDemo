function synchronize!(state::CascadeState, parameters)
    return
end

function apply_update!(state::CascadeState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

function copy_state!(state::CascadeState)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::CascadeState)
    copyto!(state.S, state.Sold)
    return
end

function residual!(
    linearsolver::LinearSolver,
    state::CascadeState,
    cascade::BucketCascade,
    Δt,
)
    S = state.S
    Sold = state.Sold
    F = linearsolver.rhs
    p_rate = state.forcing[1]
    e_rate = state.forcing[2]
    q_upstream = 0.0
    for (i, bucket) in enumerate(cascade.buckets)
        # Right-hand-side: water balance residual
        q = smooth_flow(bucket, S[i])
        F[i] =
            precipitation(bucket, p_rate) - smooth_evaporation(bucket, S[i], e_rate) - q +
            q_upstream - (S[i] - Sold[i]) / Δt
        q_upstream = q
    end
    # Newton-Raphson uses the negative residual
    @. linearsolver.rhs = -F
    return
end

function jacobian!(
    linearsolver::LinearSolver,
    state::CascadeState,
    cascade::BucketCascade,
    Δt,
)
    S = state.S
    J = linearsolver.J
    dFdSᵢ = J.d
    dFdSᵢ₋₁ = J.dl
    # dFdSᵢ₊₁ = J.du is always zero.
    e_rate = state.forcing[2]
    dq_upstream = 0.0
    for (i, bucket) in enumerate(cascade.buckets)
        # Jacobian terms
        # Lower diagonal J[i, i-1]
        if i > 1
            dFdSᵢ₋₁[i-1] = dq_upstream
        end

        # Diagonal: J[i, i]
        dq = dsmooth_flow(bucket, S[i])
        dFdSᵢ[i] = -dq - dsmooth_evaporation(bucket, S[i], e_rate) - 1.0 / Δt
        dq_upstream = dq
    end
end
