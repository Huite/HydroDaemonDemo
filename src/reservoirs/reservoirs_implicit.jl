function force!(state::CascadeState, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end

function synchronize!(state::CascadeState, parameters)
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

function residual!(solver, state::CascadeState, parameters::BucketCascade, Δt)
    S = state.S
    Sold = state.Sold
    r = solver.linearsolver.rhs
    for (i, bucket) in enumerate(parameters.buckets)
        # Right-hand-side: water balance residual
        r[i] = (
            precipitation(bucket, p_rate) - smooth_evap_cushion(bucket, S[i], e_rate)  # Use smooth version
            -
            smooth_flow(bucket, S[i])  # Use smooth version
            - (S[i] - Sold[i]) / Δt
        )
    end
    return
end

function jacobian!(solver, state::CascadeState, parameters::BucketCascade, Δt)
    S = state.S
    J = solver.linearsolver.J
    r = solver.linearsolver.rhs
    for (i, bucket) in enumerate(cascade.buckets)
        # Jacobian terms
        # Lower diagonal (element at row i, column i-1)
        if i > 1
            up = cascade.buckets[i-1]
            r[i] += smooth_flow(up, S[i-1])  # Use smooth version
            J.dl[i-1] = dsmooth_flow(up, S[i-1])  # Use smooth derivative
        end

        # Diagonal (element at row i, column i)
        J.d[i] =
            -dsmooth_flow(bucket, S[i]) - dsmooth_evap_cushion(bucket, S[i], e_rate) -
            1.0 / Δt  # Include evaporation derivative

        # Upper diagonal (element at row i, column i+1)
        if i < n_bucket
            J.du[i] = 0
        end
    end
end
