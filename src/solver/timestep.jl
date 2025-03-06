struct FixedTimeStepper
    Δt::Float64
end

function compute_time_step(timestepper, _, converged, _)
    if !converged
        error("Failed to converge. Consider using adaptive time stepping.")
    end
    return timestepper.Δt
end

struct AdaptiveTimeStepper
    Δt0::Float64
    n_increase::Int
    increase::Float64
    n_decrease::Int
    decrease::Float64
    failure::Float64
end

function compute_time_step(timestepper, Δt, converged, n_newton_iter)
    if !converged
        return Δt * timestepper.failure
    elseif n_newton_iter > timestepper.n_decrease
        return Δt * timestepper.decrease
    elseif n_newton_iter < timestepper.n_increase
        return Δt * timestepper.increase
    else
        return Δt
    end
end

function time_step!(timestepper, solver, state, parameters, Δt)
    converged, n_newton_iter = solve!(solver, state, parameters, Δt)
    while !converged
        Δt = compute_time_step(timestepper, Δt, converged, n_newton_iter)
        rewind!(state)
        converged, n_newton_iter = solve!(solver, state, parameters, Δt)
    end
    return Δt
end
