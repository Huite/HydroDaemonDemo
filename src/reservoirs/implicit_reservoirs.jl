struct NewtonSolver
    maxiter::Int
    tolerance::Float64
    alpha::Float64
    J::Tridiagonal{Float64,Vector{Float64}}
    r::Vector{Float64}
    Sold::Vector{Float64}
end

function NewtonSolver(cascade::BucketCascade, maxiter, tolerance, alpha)
    n_bucket = length(cascade.buckets)
    J = Tridiagonal(
        zeros(n_bucket - 1),  # lower diagonal
        zeros(n_bucket),  # diagonal
        zeros(n_bucket - 1),  # upper diagonal
    )
    r = zeros(n_bucket)
    Sold = zeros(n_bucket)
    return NewtonSolver(maxiter, tolerance, alpha, J, r, Sold)
end

function converged(solver::NewtonSolver)
    maxresidual = maximum(abs(residual) for residual in solver.r)
    return maxresidual < solver.tolerance
end

function residual!(cascade::BucketCascade, solver, p_rate, e_rate)
    S = cascade.S
    Sold = solver.Sold
    r = solver.r
    for (i, bucket) in enumerate(cascade.buckets)
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
