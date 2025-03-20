"""
Use Thomas for efficiency, LU for stability.
LDLT is omitted, since the Newton Jacobian is not symmetric.
"""

abstract type LinearSolver end

"""Tridiagonal linear solver."""
struct LinearSolverThomas <: LinearSolver
    n::Int
    J::Tridiagonal{Float,Vector{Float}}
    rhs::Vector{Float}
    ϕ::Vector{Float}
    γ::Vector{Float}
    β::Vector{Float}
end

function LinearSolverThomas(n)
    J = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
    return LinearSolverThomas(n, J, zeros(n), zeros(n), zeros(n), zeros(n))
end

"""Thomas algorithm."""
function linearsolve!(solver::LinearSolverThomas)
    (; n, J, rhs, ϕ, γ, β) = solver

    β = J.d[1]
    ϕ[1] = rhs[1] / β

    for j = 2:n
        γ[j] = J.du[j-1] / β
        β = J.d[j] - J.dl[j-1] * γ[j]
        if abs(β) < 1.e-12
            # This should only happen on last element of forward pass for problems
            # with zero eigenvalue. In that case the algorithmn is still stable.
            error("Beta too small!")
        end
        ϕ[j] = (rhs[j] - J.dl[j-1] * ϕ[j-1]) / β
    end

    for j = 1:n-1
        k = n - j
        ϕ[k] = ϕ[k] - γ[k+1] * ϕ[k+1]
    end
    return
end

struct LinearSolverLU <: LinearSolver
    n::Int
    J::Tridiagonal{Float,Vector{Float}}
    F::LU{Float,Tridiagonal{Float,Vector{Float}}}
    rhs::Vector{Float}
    ϕ::Vector{Float}
end

function LinearSolverLU(n)
    J = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
    return LinearSolverLU(n, J, lu(J; check = false), zeros(n), zeros(n))
end

function linearsolve!(solver::LinearSolverLU)
    # TODO: seems like this allocates
    lu!(solver.F, solver.J)
    # Inplace for Tridiagonal since Julia 1.11.
    # Note: stores result in B, overwrites diagonals.
    ldiv!(solver.F, solver.rhs)
    copyto!(solver.ϕ, solver.rhs)
end
