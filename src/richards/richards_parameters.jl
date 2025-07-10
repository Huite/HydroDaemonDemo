abstract type AbstractRichards <: Parameters end

struct RichardsParameters{C,T,B} <: AbstractRichards
    constitutive::Vector{C}
    Δz::Vector{Float}
    Δz⁻¹::Vector{Float}
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    currentforcing::Vector{Float}  # P, ET
end

function RichardsParameters(constitutive, Δz, forcing, bottomboundary, topboundary)
    Δz⁻¹ = @views 2.0 ./ (Δz[1:end-1] .+ Δz[2:end])
    n = length(constitutive)
    return RichardsParameters(
        constitutive,
        Δz,
        Δz⁻¹,
        forcing,
        bottomboundary,
        topboundary,
        n,
        zeros(2),
    )
end

struct RichardsParametersDAE{C,T,B} <: AbstractRichards
    constitutive::Vector{C}
    Δz::Vector{Float}
    Δz⁻¹::Vector{Float}
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    currentforcing::Vector{Float}  # P, ET
end

function RichardsParametersDAE(constitutive, Δz, forcing, bottomboundary, topboundary)
    Δz⁻¹ = @views 2.0 ./ (Δz[1:end-1] .+ Δz[2:end])
    n = length(constitutive)
    return RichardsParametersDAE(
        constitutive,
        Δz,
        Δz⁻¹,
        forcing,
        bottomboundary,
        topboundary,
        n,
        zeros(2),
    )
end

function prepare_ode_function(
    p::RichardsParametersDAE,
    nstate,
    analytic_jacobian,
    detect_sparsity,
)
    n = Int(nstate / 2)
    if detect_sparsity
        J = jacobian_sparsity(
            (du, u) -> waterbalance_dae!(du, u, p),
            zeros(nstate),
            zeros(nstate),
            TracerSparsityDetector(),
        )
    else
        # Construct sparsity pattern prototype
        i = Int[]
        j = Int[]
        # main diagonal
        append!(i, 1:nstate)
        append!(j, 1:nstate)
        # ψ block: sub-/super-diagonals
        append!(i, 2:n)
        append!(j, 1:n-1)
        append!(i, 1:n-1)
        append!(j, 2:n)
        # lower-left block (–C)
        append!(i, n+1:2n)
        append!(j, 1:n)
        J = sparse(i, j, ones(length(i)), nstate, nstate)
    end
    M = Diagonal([ones(n); zeros(n)])
    f = ODEFunction(waterbalance!; mass_matrix = M, jac_prototype = J)
    return f
end
