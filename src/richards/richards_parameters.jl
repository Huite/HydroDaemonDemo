abstract type AbstractRichards <: Parameters end

# [core]
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

function prepare_problem(
    parameters::RichardsParametersDAE,
    savedresults,
    nstate,
    nflow,
    detect_sparsity,
    initial,
    tspan,
)
    nunknown = nstate * 2 + nflow
    # ignores detect_sparsity; always runs SparseConnectivityTracer.
    J = jacobian_sparsity(
        (du, u) -> waterbalance_dae!(du, u, parameters),
        zeros(nunknown),
        zeros(nunknown),
        TracerSparsityDetector(),
    )
    M = Diagonal([ones(nstate); zeros(nstate); ones(nflow)])
    f = ODEFunction(waterbalance!; mass_matrix = M, jac_prototype = J)
    u0 = zeros(nunknown)
    θ0 = moisture_content.(initial, parameters.constitutive)
    @views u0[1:nstate] .= initial
    @views u0[(nstate+1):(nstate*2)] .= θ0
    params = DiffEqParams(parameters, savedresults)
    problem = ODEProblem(f, u0, tspan, params)
    return problem
end
