@kwdef struct CustomController <: AbstractController
    n_increase::Int = 5
    increase::Float64 = 1.25
    n_decrease::Int = 15
    decrease::Float64 = 0.9
    failure::Float64 = 0.5
    dtmin::Float64 = 1e-6
end

# Required interface method for step size control
function stepsize_controller!(integrator, controller::CustomController, alg)
    # Get the current algorithm's statistics
    # For implicit methods, we need to access Newton iteration info

    # Check if the step was successful
    if integrator.sol.retcode == :Success || integrator.accept_step
        # Step succeeded - check Newton iterations if available
        n_newton_iter = get_newton_iterations(integrator, alg)

        if n_newton_iter < controller.n_increase
            # Few iterations - increase step size
            new_dt = integrator.dt * controller.increase
        elseif n_newton_iter > controller.n_decrease
            # Many iterations - decrease step size
            new_dt = integrator.dt * controller.decrease
        else
            # Acceptable number of iterations - keep step size
            new_dt = integrator.dt
        end
    else
        # Step failed - reduce step size significantly
        new_dt = integrator.dt * controller.failure
        integrator.force_stepfail = true
    end

    # Enforce minimum step size
    if new_dt < controller.dtmin
        error("Time step below dtmin: $(new_dt) < $(controller.dtmin)")
    end

    # Set the new step size
    integrator.dt = new_dt
    integrator.dtcache = new_dt

    return nothing
end

# These methods are required by the AbstractController interface
function accept_step_controller(integrator, controller::CustomController)
    # Called when a step is accepted - can be used for bookkeeping
    return true
end

function reject_step_controller(integrator, controller::CustomController)
    # Called when a step is rejected - can be used for bookkeeping
    return true
end

# Additional required methods for step control
function step_accept_controller!(integrator, controller::CustomController, alg, q)
    # Called when accepting a step - handle step size update
    # Return the step size scaling factor
    if q === nothing
        # Use our custom Newton-based logic
        n_newton_iter = get_newton_iterations(integrator, alg)

        if n_newton_iter < controller.n_increase
            # Few iterations - increase step size
            return controller.increase
        elseif n_newton_iter > controller.n_decrease
            # Many iterations - decrease step size  
            return controller.decrease
        else
            # Acceptable number of iterations - keep step size
            return 1.0
        end
    else
        # Use the algorithm's computed scaling factor
        return q
    end
end

function step_reject_controller!(integrator, controller::CustomController, alg)
    # Called when rejecting a step - return aggressive reduction factor
    # Note: ODE version takes only 3 arguments (no q parameter)
    return controller.failure
end

# Helper function to extract Newton iteration count
function get_newton_iterations(integrator, alg)
    # This depends on the specific algorithm being used
    # For Rosenbrock methods, check if stats are available
    if hasfield(typeof(integrator), :stats) &&
       hasfield(typeof(integrator.stats), :nnonliniter)
        return integrator.stats.nnonliniter
    elseif hasfield(typeof(integrator.cache), :nl_iters)
        return integrator.cache.nl_iters
    elseif hasfield(typeof(integrator.cache), :newton_iters)
        return integrator.cache.newton_iters
    else
        # Fallback: use a reasonable default or try to estimate
        # from other available metrics
        if hasfield(typeof(integrator), :stats) && hasfield(typeof(integrator.stats), :nf)
            # Rough estimate: each Newton iteration typically requires 1-2 function evaluations
            return max(1, integrator.stats.nf ÷ 2)
        else
            return 5  # Default assumption
        end
    end
end
