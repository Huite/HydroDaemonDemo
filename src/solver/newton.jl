""" 
    Non-linear Newton-Raphson solver, with options for backtracking/line search
    and pseudotransient continuation.

Requires a state type with the following associated methods:

* synchronize!
* residual!
* jacobian!
* copy_state!

"""
struct NewtonSolver{LS<:LinearSolver,BT<:OptionalLineSearch,PT<:OptionalPTC}
    linearsolver::LS
    backtracking::BT
    pseudotransient::PT
    maxiter::Int
    tolerance::Float
    function NewtonSolver(;
        linearsolver::LS,
        backtracking::BT = nothing,
        pseudotransient::PT = nothing,
        maxiter::Int = 100,
        tolerance::Float64 = 1e-6,
    ) where {LS<:LinearSolver,BT<:OptionalLineSearch,PT<:OptionalPTC}
        return new{LS,BT,PT}(
            linearsolver,
            backtracking,
            pseudotransient,
            maxiter,
            tolerance,
        )
    end
end

function converged(newton::NewtonSolver)
    maxresidual = maximum(abs(r) for r in newton.linearsolver.rhs)
    return maxresidual < newton.tolerance
end

function solve!(newton::NewtonSolver{LS,BT,Nothing}, state, parameters, Δt) where {LS,BT}
    # Maintain old state for time stepping.
    copy_state!(state)
    # Synchronize dependent variables.
    synchronize!(state, parameters)

    for i = 1:newton.maxiter
        # Formulate and compute the residual.
        # Check the residual for convergence.
        residual!(newton.linearsolver, state, parameters, Δt)
        if converged(newton)
            return true, i
        end
        jacobian!(newton.linearsolver, state, parameters, Δt)
        linearsolve!(newton.linearsolver)
        apply_update!(state, newton.linearsolver, 1.0)
        #linesearch!(newton.backtracking, newton.linearsolver, state, parameters, Δt)
        synchronize!(state, parameters)
    end
    return false, newton.maxiter
end

function pseudo_timestep!(newton, state)
    ptc = newton.pseudotransient
    apply_ptc!(ptc.method, newton.linearsolver, ptc.stepselection.Δt)
    linearsolve!(newton.linearsolver)
    linesearch!(newton.backtracking, newton.linearsolver, state, parameters, Δt)
    ptc_success = check_ptc!(newton.pseudotransient, state)
    return ptc_success
end

function solve!(
    newton::NewtonSolver{LS,BT,PTC},
    state,
    parameters,
    Δt,
) where {LS,BT,PTC<:PseudoTransientContinuation}
    # Maintain old state for time stepping.
    copy_state!(state)
    # Synchronize dependent variables.
    synchronize!(state, parameters)

    # Initial step
    firststepsize!(newton.pseudotransient.stepselection)

    for i = 1:newton.maxiter
        # Formulate and compute the residual.
        # Check the residual for convergence.
        residual!(newton.linearsolver, state, parameters, Δt)
        if converged(newton)
            return true, i
        end
        jacobian!(newton.linearsolver, state, parameters, Δt)

        # Keep trying smaller time steps until we get a plausible answer.
        # Generally needs a single iteration.
        ptc_success = false
        while !ptc_success
            ptc_success = pseudo_timestep!(newton, state)
        end

        # Synchronize dependent variables.
        synchronize!(state, parameters)

        # Compute new step size based on residual evolution.
        stepsize!(newton.pseudotransient.stepselection, state, newton.linearsolver.rhs)
    end
    return false, i
end

