struct ImplicitCascadeState
    S::Vector{Float64}
    S_old::Vector{Float64}
    forcing::Vector{Float64}
end

function primary(state::ImplicitCascadeState)
    return state.S
end

struct CascadeBuckets{B,F}
    buckets::Vector{B}
    forcing::F
end

function force!(state::ImplicitCascadeState, parameters::CascadeBuckets, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end

function synchronize!(state::ImplicitCascadeState, parameters)
    return
end

function copy_state!(state::ImplicitCascadeState)
    copyto!(state.S_old, state.S)
    return
end

function rewind!(state::ImplicitCascadeState)
    copyto!(state.S, state.S_old)
    return
end

function residual!(state::ImplicitCascadeState, parameters, Δt)
    S = cascade.S
    Sold = solver.Sold
    r = solver.r
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

function jacobian!(cascade, solver, Δt)
    S = cascade.S
    J = solver.J
    r = solver.r
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

function implicit_time_step!(cascade::BucketCascade, solver::NewtonSolver, t, Δt)
    p_rate = find_rate(cascade.precipitation, t)
    e_rate = find_rate(cascade.evaporation, t)
    copyto!(solver.Sold, cascade.S)
    for jj = 1:solver.maxiter
        # Formulate the system of equations
        residual!(cascade, solver, p_rate, e_rate)
        jacobian!(cascade, solver, Δt)
        if converged(solver)
            println("Converged after iteration:", jj)
            return
        end
        S .+= ((J \ -r) .* solver.alpha)
    end
    error("Failed to converge.")
end

function implicit_run!(
    cascade::BucketCascade,
    solver::NewtonSolver,
    tstart,
    tend,
    Δt;
    saveat = nothing,
)
    if isnothing(saveat)
        saveat = Δt
    end
    save_every = Int(round(saveat / Δt))
    n_saves = Int(floor((tend - tstart) / saveat))
    n_timesteps = Int(floor((tend - tstart) / Δt))
    t = tstart

    out = zeros(length(cascade.buckets), n_saves + 1)
    out[:, 1] = cascade.S
    save_index = 2
    for i = 1:n_timesteps
        implicit_time_step!(cascade, solver, t, Δt)
        if i % save_every == 0
            out[:, save_index] = cascade.S
            save_index += 1
        end
        t += Δt
    end
    return out
end
