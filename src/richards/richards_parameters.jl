abstract type AbstractRichards <: Parameters end

struct RichardsParameters{C,T,B} <: AbstractRichards
    constitutive::Vector{C}
    Δz::Float64
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    currentforcing::Vector{Float64}  # P, ET

    function RichardsParameters(constitutive, Δz, forcing, bottomboundary, topboundary)
        new{eltype(constitutive),typeof(topboundary),typeof(bottomboundary)}(
            constitutive,
            Δz,
            forcing,
            bottomboundary,
            topboundary,
            length(constitutive),
            zeros(Float64, 2),
        )
    end
end

function Base.show(io::IO, rp::AbstractRichards)
    C = eltype(rp.constitutive)
    T = typeof(rp.topboundary)
    B = typeof(rp.bottomboundary)

    # Get clean type names
    rp_name = string(Base.typename(typeof(rp)).name)
    c_name = string(Base.typename(C).name)
    t_name = string(Base.typename(T).name)
    b_name = string(Base.typename(B).name)

    println(io, "$rp_name{$c_name,$t_name,$b_name}:")
    println(io, "  Grid: $(rp.n) layers, Δz = $(rp.Δz)")
    println(io, "  Constitutive: $(c_name)")
    println(io, "  Bottom boundary: ", rp.bottomboundary)
    println(io, "  Top boundary: ", rp.topboundary)
    println(io, "  Meteorological forcing: $(length(rp.forcing.t)) time steps")
    print(
        io,
        "  Current forcing: P = $(rp.currentforcing[1]), ET = $(rp.currentforcing[2])",
    )
end

struct RichardsParametersDAE{C,T,B} <: AbstractRichards
    constitutive::Vector{C}
    Δz::Float64
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    currentforcing::Vector{Float64}  # P, ET

    function RichardsParametersDAE(constitutive, Δz, forcing, bottomboundary, topboundary)
        new{eltype(constitutive),typeof(topboundary),typeof(bottomboundary)}(
            constitutive,
            Δz,
            forcing,
            bottomboundary,
            topboundary,
            length(constitutive),
            zeros(Float64, 2),
        )
    end
end


# Simple conversion function
function RichardsParametersDAE(p::RichardsParameters)
    return RichardsParametersDAE(
        p.constitutive,
        p.Δz,
        p.forcing,
        p.bottomboundary,
        p.topboundary,
    )
end

function reset!(p::RichardsParametersDAE, u0, initial)
    @views u0[1:p.n] .= initial
    return
end

function prepare_ode_function(p::RichardsParametersDAE, nstate, detect_sparsity)
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
