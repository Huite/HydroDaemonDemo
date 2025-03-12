struct FixedTimeStepper <: TimeStepper
    Δt0::Float
end

function compute_time_step(timestepper::FixedTimeStepper, _, converged, _)
    if !converged
        error("Failed to converge. Consider using adaptive time stepping.")
    end
    return timestepper.Δt0
end

struct AdaptiveTimeStepper <: TimeStepper
    Δt0::Float
    n_increase::Int
    increase::Float
    n_decrease::Int
    decrease::Float
    failure::Float
end

"""
Modify time step based on convergence behavior.
"""
function compute_time_step(timestepper::AdaptiveTimeStepper, Δt, converged, n_newton_iter)
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
