import HydroDaemonDemo as HDD
using DifferentialEquations
using Distributions
using DataFrames
using Random
using Plots
using BenchmarkTools
using CSV

Random.seed!(2806)

function realizations(; bounds, n)
    field_names = fieldnames(typeof(bounds))
    samples = Vector{NamedTuple{field_names,NTuple{length(field_names),Float64}}}(undef, n)
    for i = 1:n
        sample_values = map(field_names) do param_name
            param_bounds = getfield(bounds, param_name)
            rand() * (param_bounds.upper - param_bounds.lower) + param_bounds.lower
        end
        samples[i] = NamedTuple{field_names}(sample_values)
    end
    return samples
end

function explicit_model(fuse::HDD.Fuse070Parameters, tspan)
    return HDD.ExplicitHydrologicalModel(
        fuse,
        zeros(2),
        tspan,
        nothing,
        HDD.FixedTimeStepper(1.0),
    )
end

function implicit_model(fuse::HDD.Fuse070Parameters, tspan, solver, timestepper)
    return HDD.ImplicitHydrologicalModel(
        fuse,
        zeros(2),
        solver,
        tspan,
        nothing,
        timestepper,
    )
end

function diffeq_model(fuse::HDD.Fuse070Parameters, tspan, solverconfig)
    return HDD.DiffEqHydrologicalModel(fuse, zeros(2), tspan, nothing, solverconfig)
end

struct FuseBenchmarkResult{M}
    model::M
    timing::Float64
    rmse::Float64
    fidelity::Float64
end

function compute_routedflows(model)
    # Saved flows include a flow of 0 at t=0.
    parameters = HDD.get_parameters(model)
    q = diff(model.savedflows[2, :])
    routedflows = HDD.route_flows(q, parameters.μτ)
    return routedflows
end

function benchmark_fuse_model!(model, streamflow::DataFrame, refflow::DataFrame)
    HDD.run!(model)
    # Saved flows include a flow of 0 at t=0.
    routedflows = compute_routedflows(model)
    error_scheme = routedflows - streamflow.discharge
    error_exact = refflow.discharge - streamflow.discharge

    # Skip spin-up period (first 274 days)
    rmse_scheme = sqrt(mean(skipmissing(error_scheme[275:end] .^ 2)))
    rmse_exact = sqrt(mean(skipmissing(error_exact[275:end] .^ 2)))
    fidelity = rmse_scheme - rmse_exact

    initial = zeros(2)
    timing = @elapsed HDD.reset_and_run!(model, initial)

    return FuseBenchmarkResult(model, timing, rmse_scheme, fidelity)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.ExplicitPreset,
    streamflow::DataFrame,
    refflow::DataFrame,
)
    model = explicit_model(fuse, (0.0, 1523.0))
    return benchmark_fuse_model!(model, streamflow, refflow)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.ImplicitNewtonSolverPreset,
    streamflow::DataFrame,
    refflow::DataFrame,
)
    solver = HDD.NewtonSolver(
        HDD.LinearSolverThomas(2),
        relax = preset.relax,
        abstol = preset.abstol,
        reltol = preset.reltol,
    )
    model = implicit_model(fuse, (0.0, 1523.0), solver, preset.timestepper)
    return benchmark_fuse_model!(model, streamflow, refflow)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.DiffEqSolverPreset,
    streamflow::DataFrame,
    refflow::DataFrame,
)
    model = diffeq_model(fuse, (0.0, 1523.0), preset.solverconfig)
    return benchmark_fuse_model!(model, streamflow, refflow)
end

function run_reference(parameters, preset)
    models = []
    for fuse in parameters
        model = diffeq_model(fuse, (0.0, 1523.0), preset.solverconfig)
        HDD.run!(model)
        push!(models, model)
    end
    return models
end

function create_reference_streamflow(refmodels)
    refflows = []
    for model in refmodels
        routedflows = DataFrame(:discharge => compute_routedflows(model))
        push!(refflows, routedflows)
    end
    return refflows
end

function run(parameters, presets, streamflow, refflows)
    results = []
    milestone = 0  # Print every 10%
    n = length(parameters)

    for (i, fuse) in enumerate(parameters)
        parametersetresults = []
        for preset in presets
            result = benchmark(fuse, preset, streamflow, refflows[i])
            push!(parametersetresults, result)
        end
        push!(results, parametersetresults)

        percent = (i / n) * 100
        if percent >= milestone
            print("$(milestone)%...")
            milestone += 10
        end
    end
    return results
end

function create_fidelity_df(results, presets)
    # Extract fidelity data for each solver
    fidelities =
        [[parameterset[i].fidelity for parameterset in results] for i = 1:length(presets)]
    solvernames = [Symbol(HDD.name(preset)) for preset in presets]
    return DataFrame(fidelities, solvernames; makeunique = true)
end

function create_timing_df(results, presets)
    # Extract fidelity data for each solver
    timings =
        [[parameterset[i].timing for parameterset in results] for i = 1:length(presets)]
    solvernames = [Symbol(HDD.name(preset)) for preset in presets]
    return DataFrame(timings, solvernames; makeunique = true)
end

function create_timing_stats_df(results, presets)
    timings = [parameterset[i].timing for i = 1:length(presets), parameterset in results]
    solvernames = [HDD.name(preset) for preset in presets]
    # Create DataFrame with statistics (converted to milliseconds)
    df = DataFrame(
        "Method" => solvernames,
        "mean (ms)" => [1000.0 * mean(timings[i, :]) for i = 1:length(presets)],
        "median (ms)" => [1000.0 * median(timings[i, :]) for i = 1:length(presets)],
        "st. dev. (ms)" => [1000.0 * std(timings[i, :]) for i = 1:length(presets)],
    )
    return df
end

bounds = (
    S1max = (lower = 25.0, upper = 500.0),
    S2max = (lower = 50.0, upper = 5000.0),
    ϕtens = (lower = 0.050, upper = 0.950),
    r1 = (lower = 0.050, upper = 0.950),
    ku = (lower = 0.010, upper = 1000.0),
    c = (lower = 1.0, upper = 20.0),
    ki = (lower = 0.010, upper = 1000.0),
    ks = (lower = 0.001, upper = 1000.0),
    n = (lower = 1.0, upper = 10.0),
    v = (lower = 0.001, upper = 0.250),
    Acmax = (lower = 0.050, upper = 0.950),
    b = (lower = 0.001, upper = 3.0),
    μτ = (lower = 0.010, upper = 5.0),
)

samples = realizations(bounds = bounds, n = 10_000)
streamflow = HDD.read_mahurangi_streamflow("data/Mahurangi/6806 mean daily flows.csv")
forcingdf = HDD.create_mahurangi_forcingdf(
    "data/mahurangi/*daily rainfall.csv",
    "data/1340__Evaporation__daily/1340__Evaporation__Priestly-Taylor-PET__daily.csv",
)
forcing = HDD.MeteorologicalForcing(
    0.0:1.0:(length(forcingdf.Date)-1),
    forcingdf.Precipitation,
    forcingdf.Evaporation,
)
parameters = [
    HDD.Fuse070Parameters(
        ϕtens = sample.ϕtens,
        S1max = sample.S1max,
        b = sample.b,
        ku = sample.ku,
        c = sample.c,
        v = sample.v,
        μτ = sample.μτ,
        forcing = forcing,
    ) for sample in samples
]

solver_presets = [
    HDD.ExplicitPreset(),
    HDD.ImplicitNewtonSolverPreset(
        relax = HDD.SimpleLineSearch(),
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 0.5, Δtmax = 0.5),
        abstol = 1e-3,
        reltol = 1e-3,
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Tsit5(), abstol = 1e-3, reltol = 1e-3)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Rodas5P(), abstol = 1e-3, reltol = 1e-3)),
    HDD.DiffEqSolverPreset(
        HDD.SolverConfig(alg = AutoTsit5(Rodas5P()), abstol = 1e-3, reltol = 1e-3),
    ),
]

preset =
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Tsit5(), abstol = 1e-6, reltol = 1e-6))
reference_models = run_reference(parameters, preset)
refflows = create_reference_streamflow(reference_models)
results = run(parameters, solver_presets, streamflow, refflows);

fidelity_df = create_fidelity_df(results, solver_presets)
timing_df = create_timing_df(results, solver_presets)
timing_stats = create_timing_stats_df(results, solver_presets)

CSV.write("cases/output/fidelity_data.csv", fidelity_df)
CSV.write("cases/output/timing_data.csv", timing_df)
CSV.write("cases/output/fuse_timing.csv", timing_stats)
