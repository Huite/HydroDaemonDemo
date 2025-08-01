"""
Use Thomas for efficiency, LU for stability.
LDLT is omitted, since the Newton Jacobian is not symmetric.
"""

abstract type LinearSolver end

"""Tridiagonal linear solver."""
struct LinearSolverThomas <: LinearSolver
    n::Int
    J::Tridiagonal{Float64,Vector{Float64}}
    rhs::Vector{Float64}
    ϕ::Vector{Float64}
    y::Vector{Float64}
    B::Vector{Float64}
    function LinearSolverThomas(n)
        J = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
        new(n, J, zeros(n), zeros(n), zeros(n), zeros(n))
    end
end

function Base.show(io::IO, solver::LinearSolverThomas)
    print(io, "LinearSolverThomas(n=$(solver.n))")
end

"""Thomas algorithm."""
function linearsolve!(solver::LinearSolverThomas)
    (; n, J, rhs, ϕ, y, B) = solver

    B = J.d[1]
    ϕ[1] = rhs[1] / B

    for j = 2:n
        y[j] = J.du[j-1] / B
        B = J.d[j] - J.dl[j-1] * y[j]
        if abs(B) < 1.e-12
            # This should only happen on last element of forward pass for problems
            # with zero eigenvalue. In that case the algorithmn is still stable.
            error("Beta too small!")
        end
        ϕ[j] = (rhs[j] - J.dl[j-1] * ϕ[j-1]) / B
    end

    for j = 1:n-1
        k = n - j
        ϕ[k] = ϕ[k] - y[k+1] * ϕ[k+1]
    end
    return
end

struct LinearSolverLU <: LinearSolver
    n::Int
    J::Tridiagonal{Float64,Vector{Float64}}
    F::LU{Float64,Tridiagonal{Float64,Vector{Float64}}}
    rhs::Vector{Float64}
    ϕ::Vector{Float64}
    function LinearSolverLU(n)
        J = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
        new(n, J, lu(J; check = false), zeros(n), zeros(n))
    end
end

function Base.show(io::IO, solver::LinearSolverLU)
    print(io, "LinearSolverLU(n=$(solver.n))")
end

function linearsolve!(solver::LinearSolverLU)
    # TODO: seems like this allocates
    lu!(solver.F, solver.J)
    # Inplace for Tridiagonal since Julia 1.11.
    # Note: stores result in B, overwrites diagonals.
    ldiv!(solver.F, solver.rhs)
    copyto!(solver.ϕ, solver.rhs)
end
