@kwdef struct InitialHydrostatic
    watertable::Float64
end

@kwdef struct InitialConstant
    ψ::Float64
end

function initialψ(I::InitialHydrostatic, Δz, Δztotal, n)
    z = collect(Δz:Δz:Δztotal)  # heights from bottom
    return I.watertable .- z
end

function initialψ(I::InitialConstant, Δz, Δztotal, n)
    return fill(I.ψ, n)
end

struct RichardsCase
    parameters::RichardsParameters
    ψ0::Vector{Float64}
    tspan::Tuple{Float64,Float64}
end

function RichardsCase(; soil, Δz, Δztotal, tend, ψ0, bottomboundary, topboundary, forcing)
    n = Int(Δztotal / Δz)
    if isnothing(forcing)
        forcing = HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0])
    end
    return RichardsCase(
        RichardsParameters(
            constitutive = fill(soil, n),
            Δz = Δz,
            forcing = forcing,
            bottomboundary = bottomboundary,
            topboundary = topboundary,
        ),
        initialψ(ψ0, Δz, Δztotal, n),
        (0.0, tend),
    )
end

function implicit_model(case::RichardsCase, solver, timestepper, saveat)
    return HydroDaemonDemo.ImplicitHydrologicalModel(
        case.parameters,
        case.ψ0,
        solver,
        case.tspan,
        saveat,
        timestepper,
    )
end

function diffeq_model(case::RichardsCase, solverconfig, saveat)
    return HydroDaemonDemo.DiffEqHydrologicalModel(
        case.parameters,
        case.ψ0,
        case.tspan,
        saveat,
        solverconfig,
    )
end

function diffeq_model_dae(case::RichardsCase, solverconfig, saveat)
    θ0 = HydroDaemonDemo.moisture_content.(case.ψ0, case.parameters.constitutive)
    u0 = [case.ψ0; θ0]
    return HydroDaemonDemo.DiffEqHydrologicalModel(
        HydroDaemonDemo.RichardsParametersDAE(case.parameters),
        u0,
        case.tspan,
        saveat,
        solverconfig,
    )
end

function storage(ψ, parameters::RichardsParameters)
    θ = moisture_content.(ψ, parameters.constitutive)
    S_elastic = [c.Ss * θi / c.θs for (c, θi) in zip(parameters.constitutive, θ)]
    return parameters.Δz * (θ + S_elastic)
end

function waterbalance_error(model::DiffEqHydrologicalModel)
    flows = model.saved[]

end

struct BenchMarkResult{M}
    model::M
    trial::BenchmarkTools.Trial
    waterbalance_error::Vector{Float64}
end

function benchmark!(model, case::RichardsCase)
    trial = @benchmark HydroDaemonDemo.reset_and_run!($model, $(case.ψ0))
    return BenchMarkResult(model, trial, zeros(2))
end

function Base.show(io::IO, result::BenchMarkResult)
    println(io, "BenchmarkResult:")
    println(io, "  Model: $(typeof(result.model).name.name)")  # Just the type name
    println(io, "  Time: $(BenchmarkTools.prettytime(minimum(result.trial).time))")
    println(io, "  Memory: $(BenchmarkTools.prettymemory(minimum(result.trial).memory))")
    println(io, "  Allocations: $(minimum(result.trial).allocs)")
    if !isempty(result.waterbalance_error)
        println(io, "  Water balance error: $(result.waterbalance_error)")
    end
end
