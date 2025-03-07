""" 
    Non-linear Newton-Raphson solver, with options for backtracking/line search
    and pseudotransient continuation.

Requires a state type with the following associated methods:

* synchronize!
* residual!
* jacobian!
* copy_state!

"""
struct NewtonSolver{LS<:LinearSolver,BT<:Union{LineSearch,Nothing},PT}
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
        tolerance::Float64 = 1e-6
    ) where {LS<:LinearSolver, BT<:Union{LineSearch,Nothing}, PT}
        return new{LS, BT, PT}(linearsolver, backtracking, pseudotransient, maxiter, tolerance)
    end
end

function converged(newton::NewtonSolver)
    maxresidual = maximum(abs(r) for r in newton.linearsolver.rhs)
    return maxresidual < newton.tolerance
end

function solve!(newton::NewtonSolver, state, parameters, Δt)
    # Maintain old state for time stepping.
    copy_state!(state)
    # Synchronize dependent variables.
    synchronize!(state, parameters)

    ptcΔt = initial_pseudotimestep(newton.pseudotransient, state)
    for i = 1:newton.maxiter
        # Formulate and compute the residual.
        residual!(newton, state, parameters, Δt)
        # Check the residual for convergence.
        if converged(newton)
            return true, i
        end
        # Linearize and solve.
        jacobian!(state, newton.linearsolver, parameters, Δt)

        ptc_succes = False
        while !ptc_succes
            ptcΔt = ptc!(newton.pseudotransient, state)
            linearsolve!(newton.linearsolver)
            # Find and apply optimized update.
            # TODO: if no PTC, linesearch can be also be checked for plausibility.
            linesearch!(newton, newton.backtracking, state, parameters, Δt)
            ptc_succes = check_ptc!(newton.pseudotransient, state)
        end

        # Synchronize dependent variables.
        synchronize!(state, parameters)
    end
    return false, i
end

