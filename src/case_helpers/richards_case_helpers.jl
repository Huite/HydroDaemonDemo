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
    saveat::Vector{Float64}
end

function RichardsCase(;
    soil,
    Δz,
    Δztotal,
    tend,
    dt,
    ψ0,
    bottomboundary,
    topboundary,
    forcing,
)
    n = Int(Δztotal / Δz)
    if isnothing(forcing)
        forcing = MeteorologicalForcing([0.0], [0.0], [0.0])
    end
    saveat = collect(0.0:dt:tend)
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
        saveat,
    )
end

function implicit_model(case::RichardsCase, solver, timestepper, saveat)
    return ImplicitHydrologicalModel(
        case.parameters,
        case.ψ0,
        solver,
        case.tspan,
        saveat,
        timestepper,
    )
end

function diffeq_model(case::RichardsCase, solverconfig, saveat)
    return DiffEqHydrologicalModel(
        case.parameters,
        case.ψ0,
        case.tspan,
        saveat,
        solverconfig,
    )
end

function diffeq_model_dae(case::RichardsCase, solverconfig, saveat)
    return DiffEqHydrologicalModel(
        RichardsParametersDAE(case.parameters),
        case.ψ0,
        case.tspan,
        saveat,
        solverconfig,
    )
end

function storage(ψ, parameters::AbstractRichards)
    θ = moisture_content.(ψ, parameters.constitutive)
    S_elastic = [c.Ss * θi / c.θs for (c, θi) in zip(parameters.constitutive, θ)]
    return parameters.Δz * (θ + S_elastic)
end

function waterbalance_dataframe(model)
    # Compute cumulative flows back to reporting steps.
    # One smaller (vertex vs interval): no cumulative yet flow at t=0.
    parameters = get_parameters(model)
    n = parameters.n
    S = [storage(ψ, parameters) for ψ in eachcol(model.saved[1:n, :])]
    return DataFrame(
        :t => model.saveat,
        :storage => sum.(S),
        :qbot => model.savedflows[1, :],
        :qtop => model.savedflows[2, :],
    )
end

function massbalance_bias(waterbalance)
    ΔS = waterbalance.storage[end] - waterbalance.storage[1]
    error = waterbalance.qbot[end] + waterbalance.qtop[end] - ΔS
    return error
end


function massbalance_rmse(waterbalance)
    ΔS = diff(waterbalance.storage)
    qb = diff(waterbalance.qbot)
    qt = diff(waterbalance.qtop)
    error = qt + qb - ΔS
    return sqrt(mean(error .^ 2))
end

struct BenchMarkResult{M}
    model::M
    trial::BenchmarkTools.Trial
    waterbalance::DataFrame
    mass_bias::Float64
    mass_rsme::Float64
end

function benchmark_model!(model, case::RichardsCase)
    run!(model)
    trial = @benchmark reset_and_run!($model, $(case.ψ0)) samples=20
    #time = @elapsed reset_and_run!(model, case.ψ0)
    wb = waterbalance_dataframe(model)
    return BenchMarkResult(model, trial, wb, massbalance_bias(wb), massbalance_rmse(wb))
end

function Base.show(io::IO, result::BenchMarkResult)
    println(io, "BenchmarkResult:")
    println(io, "  Model: $(typeof(result.model).name.name)")  # Just the type name
    println(io, "  Time: $(BenchmarkTools.prettytime(minimum(result.trial).time))")
    #    println(io, "  Time: $(result.time)")F
    println(io, "  Memory: $(BenchmarkTools.prettymemory(minimum(result.trial).memory))")
    println(io, "  Allocations: $(minimum(result.trial).allocs)")
    println(io, "  Mass Bias: $(result.mass_bias)")
    print(io, "  Mass RMSE: $(result.mass_rsme)")
end


struct ExplicitPreset end

function name(preset::ExplicitPreset)
    return "Explicit"
end

abstract type ImplicitSolverPreset end

@kwdef struct ImplicitNewtonSolverPreset{T,R} <: ImplicitSolverPreset
    abstol::Float64=1e-6
    reltol::Float64=1e-6
    relax::R=ScalarRelaxation(0.0)
    timestepper::T
end

@kwdef struct ImplicitPicardSolverPreset{T,R} <: ImplicitSolverPreset
    abstol::Float64=1e-6
    reltol::Float64=1e-6
    relax::R=ScalarRelaxation(0.0)
    timestepper::T
end

function name(_::ImplicitNewtonSolverPreset)
    return "Implicit Newton"
end

function name(_::ImplicitPicardSolverPreset)
    return "Implicit Picard"
end

struct DiffEqSolverPreset
    solverconfig::SolverConfig
end

function name(preset::DiffEqSolverPreset)
    algname = string(typeof(preset.solverconfig.alg).name.name)
    return "DiffEq-$(algname)"
end

struct DAEDiffEqSolverPreset
    solverconfig::SolverConfig
end

function name(preset::DAEDiffEqSolverPreset)
    algname = string(typeof(preset.solverconfig.alg).name.name)
    return "DAE-DiffEq-$(algname)"
end

function benchmark!(case::RichardsCase, preset::ImplicitNewtonSolverPreset)
    implicit_solver = NewtonSolver(
        LinearSolverThomas(case.parameters.n),
        relax = preset.relax,
        abstol = preset.abstol,
        reltol = preset.reltol,
    )
    result = benchmark_model!(
        implicit_model(case, implicit_solver, preset.timestepper, case.saveat),
        case,
    )
    return result
end


function benchmark!(case::RichardsCase, preset::ImplicitPicardSolverPreset)
    implicit_solver = PicardSolver(
        LinearSolverThomas(case.parameters.n),
        relax = preset.relax,
        abstol = preset.abstol,
        reltol = preset.reltol,
    )
    result = benchmark_model!(
        implicit_model(case, implicit_solver, preset.timestepper, case.saveat),
        case,
    )
    return result
end

function benchmark!(case::RichardsCase, preset::DiffEqSolverPreset)
    result = benchmark_model!(diffeq_model(case, preset.solverconfig, case.saveat), case)
    return result
end

function benchmark!(case::RichardsCase, preset::DAEDiffEqSolverPreset)
    result =
        benchmark_model!(diffeq_model_dae(case, preset.solverconfig, case.saveat), case)
    return result
end
