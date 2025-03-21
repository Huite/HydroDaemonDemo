struct FixedTimeStepper <: TimeStepper
    Δt0::Float
end

# Called in implicit model
function compute_timestep_size(timestepper::FixedTimeStepper, _, converged::Bool, _)
    if !converged
        error("Failed to converge. Consider using adaptive time stepping.")
    end
    return timestepper.Δt0
end

# Called in explicit model
function compute_timestep_size(timestepper::FixedTimeStepper, state::State, parameters::Parameters, Δt)
    return timestepper.Δt0
end

struct AdaptiveTimeStepper <: TimeStepper
    Δt0::Float
    n_increase::Int
    increase::Float
    n_decrease::Int
    decrease::Float
    failure::Float
    function AdaptiveTimeStepper(
        Δt0;
        n_increase = 5,
        increase = 1.25,
        n_decrease = 15,
        decrease = 0.9,
        failure = 0.5,
    )
        return new(Δt0, n_increase, increase, n_decrease, decrease, failure)
    end
end

"""
Modify time step based on convergence behavior.
"""
function compute_timestep_size(
    timestepper::AdaptiveTimeStepper,
    Δt,
    converged,
    n_newton_iter,
)
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


struct CFLTimeStepper <: TimeStepper
    Δt0::Float
    target::Float
    minstep::Float
end
