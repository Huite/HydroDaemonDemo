"""
Picard iteration generally solves directly for the new y of the iterate.
However, without less of generality we can solve for the update of y instead,
similar to a Newton-Raphson method. In that case, they may share the residual
formulation.
"""
struct PicardSolver{LS<:LinearSolver,R<:Relaxation}
    linearsolver::LS
    relax::R
    maxiter::Int
    abstol::Float64
    reltol::Float64
    calls::Vector{Float64}
end

function PicardSolver(
    linearsolver::LinearSolver;
    relax::Relaxation = ScalarRelaxation(0.0),
    maxiter::Int = 500,
    abstol::Float64 = 1e-6,
    reltol::Float64 = 1e-6,
)
    return PicardSolver(linearsolver, relax, maxiter, abstol, reltol)
end

function Base.show(io::IO, solver::PicardSolver)
    LS = typeof(solver.linearsolver)
    # Get short names for the types
    ls_name = string(LS)
    print(
        io,
        "PicardSolver{$ls_name}(maxiter=$(solver.maxiter), abstol=$(solver.abstol)), reltol=$(solver.reltol),",
    )
end

function converged(picard::PicardSolver, state)
    residual = picard.linearsolver.rhs
    return all(
        i -> abs(residual[i]) < picard.abstol + picard.reltol * abs(state[i]),
        eachindex(residual),
    )
end

function setmatrix!(picard::PicardSolver, state, parameters, Δt)
    coefficients!(picard.linearsolver.J, state, parameters, Δt)
end
