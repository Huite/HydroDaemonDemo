""" 
    Non-linear Newton-Raphson solver, with options for backtracking/line search.

Requires a state type with the following associated methods:

* residual!
* jacobian!
* copy_state!

"""
struct NewtonSolver{LS<:LinearSolver,R<:Relaxation}
    linearsolver::LS
    relax::R
    maxiter::Int
    abstol::Float64
    reltol::Float64
end

function NewtonSolver(
    linearsolver::LinearSolver;
    relax::Relaxation = ScalarRelaxation(0.0),
    maxiter::Int = 100,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-6,
)
    return NewtonSolver(linearsolver, relax, maxiter, abstol, reltol)
end

function Base.show(io::IO, solver::NewtonSolver)
    LS = typeof(solver.linearsolver)
    R = typeof(solver.relax)
    # Get short names for the types
    ls_name = string(LS)
    r_name = string(R)
    print(
        io,
        "NewtonSolver{$ls_name,$r_name}(maxiter=$(solver.maxiter), abstol=$(solver.abstol)), reltol=$(solver.reltol),",
    )
end

function converged(newton::NewtonSolver, state)
    residual = newton.linearsolver.rhs
    return all(
        i -> abs(residual[i]) < newton.abstol + newton.reltol * abs(state[i]),
        eachindex(residual),
    )
end


function solve!(newton::NewtonSolver{LS,R}, state, parameters, Δt) where {LS,R}
    # Maintain old state for time stepping.
    copy_state!(state, parameters)
    # Compute initial residual
    residual!(newton.linearsolver.rhs, state, parameters, Δt)

    for i = 1:newton.maxiter
        # Check the residual immediately for convergence.
        if converged(newton, primary(state))
            return true, i
        end
        jacobian!(newton.linearsolver.J, state, parameters, Δt)
        linearsolve!(newton.linearsolver)
        # The newton step will update the state, synchronize, and recompute
        # residual.
        newton_step!(newton.relax, newton.linearsolver, state, parameters, Δt)
    end
    return false, newton.maxiter
end
