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
    solverconfig,
    initial,
    tspan,
)
    nunknown = nstate * 2 + nflow
    Jpattern = jacobian_sparsity(
        (du, u) -> waterbalance_dae!(du, u, parameters),
        zeros(nunknown),
        zeros(nunknown),
        TracerSparsityDetector(),
    )
    J = Float64.(Jpattern)
    #M = Diagonal([zeros(nstate); ones(nstate); ones(nflow)])

    Δz = parameters.Δz
    Z = spzeros(Float64, nstate, nstate)
    I_n = spdiagm(0 => fill(Δz, nstate))
    Ss = spdiagm(0 => Float64[con.Ss * Δz for con in parameters.constitutive])
    M = blockdiag([Z Z; Ss I_n], sparse(Float64, I, nflow, nflow))

    f = ODEFunction(waterbalance!; mass_matrix = M, jac_prototype = J)
    u0 = zeros(nunknown)
    θ0 = moisture_content.(initial, parameters.constitutive)
    @views u0[1:nstate] .= initial
    @views u0[(nstate+1):(nstate*2)] .= θ0
    params = DiffEqParams(parameters, savedresults)
    problem = ODEProblem(f, u0, tspan, params)
    abstol, reltol =
        create_tolvectors(nunknown, nflow, solverconfig.abstol, solverconfig.reltol)
    return problem, abstol, reltol
end

function reset!(p::RichardsParametersDAE, u0, initial)
    u0 .= 0.0
    n = p.n
    ψ0 = @view u0[1:n]
    θ0 = @view u0[(n+1):(n*2)]
    ψ0 .= initial
    for i = 1:n
        θ0[i] = moisture_content(ψ0[i], p.constitutive[i])
    end
    return
end
