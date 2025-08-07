abstract type AbstractRichards <: Parameters end

struct RichardsParameters{C,T,B} <: AbstractRichards
    constitutive::Vector{C}
    Δz::Float64
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    currentforcing::Vector{Float64}  # P, ET

    function RichardsParameters(; constitutive, Δz, forcing, bottomboundary, topboundary)
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


function create_tolvectors(nstate, nflows, abstol, reltol)
    vector_abstol = fill(abstol, nstate + nflows)
    vector_reltol = fill(reltol, nstate + nflows)
    @views vector_abstol[end-nflows:end] .= 1e12
    @views vector_reltol[end-nflows:end] .= 1e12
    return vector_abstol, vector_reltol
end

function prepare_ode_function(
    p::RichardsParameters,
    initial,
    detect_sparsity,
    abstol,
    reltol,
)
    nstate = length(initial)
    nflows = 2
    # Make room for qbottom, qtop
    if detect_sparsity
        J = jacobian_sparsity(
            (du, u) -> waterbalance!(du, u, p),
            zeros(nstate + nflows),
            zeros(nstate + nflows),
            TracerSparsityDetector(),
        )
    else
        J = Tridiagonal(
            zeros(nstate + nflows - 1),
            zeros(nstate + nflows),
            zeros(nstate + nflows - 1),
        )
    end

    f = ODEFunction(waterbalance!; jac_prototype = J)
    vabstol, vreltol = create_tolvectors(nstate, nflows, abstol, reltol)
    return f, vcat(initial, zeros(nflows)), vabstol, vreltol
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

function prepare_ode_function(p::RichardsParametersDAE, initial, detect_sparsity)
    nstate = length(initial)
    nflows = 2
    n = Int(nstate / 2)
    if detect_sparsity
        J = jacobian_sparsity(
            (du, u) -> waterbalance_dae!(du, u, p),
            zeros(nstate + nflows),
            zeros(nstate + nflows),
            TracerSparsityDetector(),
        )
    else
        # TODO: sparsity pattern for qtop, qbot!
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
        J = sparse(i, j, ones(length(i)), nstate + 2, nstate + 2)
    end
    M = Diagonal([ones(n); zeros(n); ones(nflows)])
    f = ODEFunction(waterbalance!; mass_matrix = M, jac_prototype = J)
    nflows = 2
    vabstol, vreltol = create_tolvectors(nstate, nflows, abstol, reltol)
    return f, vcat(initial, zeros(nflows)), vabstol, vreltol
end
