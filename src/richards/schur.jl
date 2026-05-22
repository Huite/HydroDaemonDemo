# Custom linear solver that implements a Schur reduction:
#
# - Improves Jacobian conditioning
# - Halves the size of the system

import LinearSolve as LS
import LinearAlgebra as LA


struct SchurCache{InnerCache,MatT,IdxT}
    inner::InnerCache
    n::Int
    C::MatT
    B::MatT
    S::MatT
    C_nzval_idx::IdxT   # indices into W.nzval for C block
    B_nzval_idx::IdxT   # indices into W.nzval for B block
    C_in_S_idx::IdxT    # indices into S.nzval for C entries
    B_in_S_idx::IdxT    # indices into S.nzval for B entries
end

struct CustomLinearSolver{InnerAlg} <: LS.SciMLLinearSolveAlgorithm
    inner_alg::InnerAlg
end

LS.needs_concrete_A(::CustomLinearSolver) = true
LS.default_alias_A(::CustomLinearSolver, ::Any, ::Any) = true
LS.default_alias_b(::CustomLinearSolver, ::Any, ::Any) = true

function block_nzval_indices(W::SparseMatrixCSC, rows::UnitRange, cols::UnitRange)
    idx = Int[]
    for j in cols
        for k = W.colptr[j]:(W.colptr[j+1]-1)
            if W.rowval[k] in rows
                push!(idx, k)
            end
        end
    end
    return idx
end

function block_in_S_indices(
    S::SparseMatrixCSC,
    src::SparseMatrixCSC,
    row_offset::Int,
    col_offset::Int,
)
    idx = Int[]

    for j = 1:size(src, 2)
        for k = src.colptr[j]:(src.colptr[j+1]-1)
            i = src.rowval[k]

            s_col = j + col_offset
            found = false

            for sk = S.colptr[s_col]:(S.colptr[s_col+1]-1)
                if S.rowval[sk] == i + row_offset
                    push!(idx, sk)
                    found = true
                    break
                end
            end

            found || error("Entry ($i, $j) from source block not found in Schur matrix")
        end
    end

    return idx
end


function LS.init_cacheval(
    alg::CustomLinearSolver,
    A,
    b,
    u,
    Pl,
    Pr,
    maxiters,
    abstol,
    reltol,
    verbose,
    assump::LS.OperatorAssumptions,
)
    A_mat = convert(SparseMatrixCSC, A)
    n = size(A_mat, 1) ÷ 2
    ψ = 1:n
    θ = (n+1):(2n)

    # Build initial Schur complement to get correct sparsity/type for inner cache
    α = A_mat[n+1, n+1]
    C = A_mat[ψ, ψ]
    B = A_mat[θ, ψ]
    S = C - (1/α) * B  # allocate once to establish sparsity

    C_nzval_idx = block_nzval_indices(A_mat, ψ, ψ)
    B_nzval_idx = block_nzval_indices(A_mat, θ, ψ)

    # C maps into S with no offset (C occupies same block as S)
    C_in_S_idx = block_in_S_indices(S, C, 0, 0)
    # B maps into S with same column range, but B's rows are offset,
    # however in S coordinates B contributes to the same rows as C
    B_in_S_idx = block_in_S_indices(S, B, 0, 0)

    b_red = zeros(eltype(b), n)
    inner = LS.init(LS.LinearProblem(S, b_red), alg.inner_alg)

    return SchurCache(inner, n, C, B, S, C_nzval_idx, B_nzval_idx, C_in_S_idx, B_in_S_idx)
end

function SciMLBase.solve!(cache::LS.LinearCache, alg::CustomLinearSolver; kwargs...)
    sc = cache.cacheval
    n = sc.n
    W = cache.A
    b = cache.b
    du = cache.u

    ψ = 1:n
    θ = (n+1):(2n)

    bψ = @view b[ψ]
    bθ = @view b[θ]
    dψ = @view du[ψ]
    dθ = @view du[θ]

    α = W[n+1, n+1]
    inv_α = 1 / α

    if cache.isfresh
        sc.C.nzval .= @view W.nzval[sc.C_nzval_idx]
        sc.B.nzval .= @view W.nzval[sc.B_nzval_idx]

        fill!(sc.S.nzval, 0)
        @inbounds for (s_idx, c_idx) in zip(sc.C_in_S_idx, eachindex(sc.C.nzval))
            sc.S.nzval[s_idx] += sc.C.nzval[c_idx]
        end
        @inbounds for (s_idx, b_idx) in zip(sc.B_in_S_idx, eachindex(sc.B.nzval))
            sc.S.nzval[s_idx] -= inv_α * sc.B.nzval[b_idx]
        end

        sc.inner.A = sc.S
        sc.inner.isfresh = true
        cache.isfresh = false
    end

    @. sc.inner.b = bψ - inv_α * bθ

    LS.solve!(sc.inner, alg.inner_alg; kwargs...)
    dψ .= sc.inner.u

    dθ .= bθ
    LA.mul!(dθ, sc.B, dψ, -1, 1)
    dθ .*= inv_α

    return SciMLBase.build_linear_solution(
        alg,
        du,
        nothing,
        cache;
        retcode = ReturnCode.Success,
    )
end
