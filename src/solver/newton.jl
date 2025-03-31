""" 
    Non-linear Newton-Raphson solver, with options for backtracking/line search
    and pseudotransient continuation.

Requires a state type with the following associated methods:

* synchronize!
* residual!
* jacobian!
* copy_state!

"""
struct NewtonSolver{LS<:LinearSolver,R<:Relaxation,PT<:OptionalPTC}
    linearsolver::LS
    relax::R
    pseudotransient::PT
    maxiter::Int
    tolerance::Float
end

function NewtonSolver(
    linearsolver::LinearSolver;
    relax::Relaxation = ScalarRelaxation(0.0),
    pseudotransient::OptionalPTC = nothing,
    maxiter::Int = 100,
    tolerance::Float = 1e-6,
)
    return NewtonSolver(linearsolver, relax, pseudotransient, maxiter, tolerance)
end

function converged(newton::NewtonSolver)
    maxresidual = maximum(abs(r) for r in newton.linearsolver.rhs)
    return maxresidual < newton.tolerance
end

function solve!(newton::NewtonSolver{LS,R,Nothing}, state, parameters, Δt) where {LS,R}
    # Maintain old state for time stepping.
    copy_state!(state)
    # Compute initial residual
    residual!(newton.linearsolver, state, parameters, Δt)

    for i = 1:newton.maxiter
        # Check the residual immediately for convergence.
        if converged(newton)
            return true, i
        end
        jacobian!(newton.linearsolver, state, parameters, Δt)
        linearsolve!(newton.linearsolver)
        # The newton step will update the state, synchronize, and recompute
        # residual.
        newton_step!(newton.relax, newton.linearsolver, state, parameters, Δt)
    end
    return false, newton.maxiter
end

function pseudotimestep!(newton, state)
    ptc = newton.pseudotransient
    apply_ptc!(ptc.method, newton.linearsolver, ptc.stepselection.Δt)
    linearsolve!(newton.linearsolver)
    newton_step!(newton.relax, newton.linearsolver, state, parameters, Δt)
    ptc_success = check_ptc!(newton.pseudotransient, state)
    return ptc_success
end

function solve!(
    newton::NewtonSolver{LS,R,PTC},
    state,
    parameters,
    Δt,
) where {LS,R,PTC<:PseudoTransientContinuation}
    # Maintain old state for time stepping.
    copy_state!(state)
    # Synchronize dependent variables.
    residual!(newton.linearsolver, state, parameters, Δt)

    # Initial step
    firststepsize!(newton.pseudotransient.stepselection)

    for i = 1:newton.maxiter
        # Formulate and compute the residual.
        # Check the residual for convergence.
        if converged(newton)
            return true, i
        end
        jacobian!(newton.linearsolver, state, parameters, Δt)

        # Keep trying smaller time steps until we get a plausible answer.
        # Generally needs a single iteration.
        ptc_success = false
        while !ptc_success
            ptc_success = pseudotimestep!(newton, state)
        end

        # Compute new step size based on residual evolution.
        stepsize!(newton.pseudotransient.stepselection, state, newton.linearsolver.rhs)
    end
    return false, i
end

