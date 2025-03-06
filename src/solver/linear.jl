"""
Use Thomas for efficiency, LU for stability.
LDLT is omitted, since the Newton Jacobian is not symmetric.
"""

abstract type LinearSolver end

const Float = Float64

"""Tridiagonal linear solver."""
struct LinearSolverThomas <: LinearSolver
    n::Int
    M::Tridiagonal{Float,Vector{Float}}
    rhs::Vector{Float64}
    ϕ::Vector{Float64}
    γ::Vector{Float64}
    β::Vector{Float64}
end

function LinearSolverThomas(n)
    M = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
    return LinearSolverThomas(n, M, zeros(n), zeros(n), zeros(n), zeros(n))
end

"""Thomas algorithm."""
function linearsolve!(solver::LinearSolverThomas)
    (; n, M, rhs, ϕ, γ, β) = solver

    β = M.d[1]
    ϕ[1] = rhs[1] / β

    for j = 2:n
        γ[j] = M.du[j-1] / β
        β = M.d[j] - M.dl[j-1] * γ[j]
        if abs(β) < 1.e-12
            # This should only happen on last element of forward pass for problems
            # with zero eigenvalue. In that case the algorithmn is still stable.
            error("Beta too small!")
            break
        end
        ϕ[j] = (rhs[j] - M.dl[j-1] * ϕ[j-1]) / β
    end

    for j = 1:n-1
        k = n - j
        ϕ[k] = ϕ[k] - γ[k+1] * ϕ[k+1]
    end
    return
end

struct LinearSolverLU <: LinearSolver
    n::Int
    M::Tridiagonal{Float,Vector{Float}}
    F::LU{Float64,Tridiagonal{Float,Vector{Float}}}
    rhs::Vector{Float64}
    ϕ::Vector{Float}
end

function LinearSolverLU(n)
    M = Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1))
    return LinearSolverLU(n, M, lu(M), zeros(n), zeros(n))
end

function linearsolve!(solver::LinearSolverLU)
    lu!(solver.F, solver.M)
    # Inplace for Tridiagonal since Julia 1.11.
    # Note: stores result in B, overwrites diagonals.
    ldiv!(solver.F, solver.rhs)
    copyto!(solver.ϕ, solver.rhs)
end
