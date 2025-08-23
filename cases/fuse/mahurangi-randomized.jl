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
end

function benchmark_fuse_model!(model, streamflow::DataFrame)
    HDD.run!(model)

    # Saved flows include a flow of 0 at t=0.
    parameters = HDD.get_parameters(model)
    q = diff(model.savedflows[2, :])
    routedflows = HDD.route_flows(q, parameters.μτ)
    error = routedflows - streamflow.discharge
    # Skip spin-up period (first 274 days)
    rmse = sqrt(mean(skipmissing(error[275:end]) .^ 2))

    initial = zeros(2)
    timing = @elapsed HDD.reset_and_run!(model, initial);
    return FuseBenchmarkResult(model, timing, rmse)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.ExplicitPreset,
    streamflow::DataFrame,
)
    model = explicit_model(fuse, (0.0, 1523.0))
    return benchmark_fuse_model!(model, streamflow)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.ImplicitSolverPreset,
    streamflow::DataFrame,
)
    solver = HDD.NewtonSolver(
        HDD.LinearSolverThomas(2),
        relax = preset.relax,
        abstol = preset.abstol,
        reltol = preset.reltol,
    )
    model = implicit_model(fuse, (0.0, 1523.0), solver, preset.timestepper)
    return benchmark_fuse_model!(model, streamflow)
end

function benchmark(
    fuse::HDD.Fuse070Parameters,
    preset::HDD.DiffEqSolverPreset,
    streamflow::DataFrame,
)
    model = diffeq_model(fuse, (0.0, 1523.0), preset.solverconfig)
    return benchmark_fuse_model!(model, streamflow)
end

function run(parameters, presets, streamflow)
    results = []
    for (i, fuse) in enumerate(parameters)
        println(i)
        parametersetresults = []
        for preset in presets
            println("benchmarking ", HDD.name(preset))
            result = benchmark(fuse, preset, streamflow)
            push!(parametersetresults, result)
        end
        push!(results, parametersetresults)
    end
    return results
end

function scatter_plots(results)
    function calculate_r2(y_actual, y_predicted)
        ss_res = sum((y_actual .- y_predicted) .^ 2)
        ss_tot = sum((y_actual .- mean(y_actual)) .^ 2)
        return 1 - (ss_res / ss_tot)
    end

    rmse_explicit = [parameterset[1].rmse for parameterset in results]
    rmse_implicit = [parameterset[2].rmse for parameterset in results]
    rmse_tsit5 = [parameterset[3].rmse for parameterset in results]
    rmse_rodas5p = [parameterset[4].rmse for parameterset in results]
    rmse_autotsit = [parameterset[5].rmse for parameterset in results]
    rmse_ref = [parameterset[6].rmse for parameterset in results]

    # Calculate R² values
    r2_explicit = calculate_r2(rmse_ref, rmse_explicit)
    r2_implicit = calculate_r2(rmse_ref, rmse_implicit)
    r2_tsit5 = calculate_r2(rmse_ref, rmse_tsit5)
    r2_rodas5p = calculate_r2(rmse_ref, rmse_rodas5p)
    r2_autotsit = calculate_r2(rmse_ref, rmse_autotsit)

    # Create plots with smaller markers, 1:1 line, and R² annotations
    p1 = scatter(
        rmse_ref,
        rmse_explicit,
        legend = false,
        markersize = 3,
        xlabel = "Exact",
        ylabel = "Explicit",
    )
    plot!(p1, [0, 20], [0, 20], line = (:black, :dash, 1), legend = false)
    annotate!(p1, 18, 2, text("R² = $(round(r2_explicit, digits=3))", 10, :right))

    p2 = scatter(
        rmse_ref,
        rmse_implicit,
        legend = false,
        markersize = 3,
        xlabel = "Exact",
        ylabel = "Implicit",
    )
    plot!(p2, [0, 20], [0, 20], line = (:black, :dash, 1), legend = false)
    annotate!(p2, 18, 2, text("R² = $(round(r2_implicit, digits=3))", 10, :right))

    p3 = scatter(
        rmse_ref,
        rmse_tsit5,
        legend = false,
        markersize = 3,
        xlabel = "Exact",
        ylabel = "Tsit5",
    )
    plot!(p3, [0, 20], [0, 20], line = (:black, :dash, 1), legend = false)
    annotate!(p3, 18, 2, text("R² = $(round(r2_tsit5, digits=3))", 10, :right))

    p4 = scatter(
        rmse_ref,
        rmse_rodas5p,
        legend = false,
        markersize = 3,
        xlabel = "Exact",
        ylabel = "Rodas5P",
    )
    plot!(p4, [0, 20], [0, 20], line = (:black, :dash, 1), legend = false)
    annotate!(p4, 18, 2, text("R² = $(round(r2_rodas5p, digits=3))", 10, :right))

    p5 = scatter(
        rmse_ref,
        rmse_autotsit,
        legend = false,
        markersize = 3,
        xlabel = "Exact",
        ylabel = "AutoTsit5",
    )
    plot!(p5, [0, 20], [0, 20], line = (:black, :dash, 1), legend = false)
    annotate!(p5, 18, 2, text("R² = $(round(r2_autotsit, digits=3))", 10, :right))

    result = plot(
        p1,
        p2,
        p3,
        p4,
        p5,
        layout = (2, 3),
        size = (800, 500),
        margin = 5Plots.mm,
        xlims = (0, 20),
        ylims = (0, 20),
    )
    return result
end

function timing_df(results)
    timing_explicit = [parameterset[1].timing for parameterset in results]
    timing_implicit = [parameterset[2].timing for parameterset in results]
    timing_tsit5 = [parameterset[3].timing for parameterset in results]
    timing_rodas5p = [parameterset[4].timing for parameterset in results]
    timing_autotsit = [parameterset[5].timing for parameterset in results]

    timing_df = DataFrame(
        "Method" => [
            "Explicit Euler",
            "Implicit Euler",
            "DiffEq-Tsit5",
            "DiffEq-Rodas5P",
            "DiffEq-AutoTsit5",
        ],
        "mean (ms)" => [
            1000.0 * mean(timing_explicit),
            1000.0 * mean(timing_implicit),
            1000.0 * mean(timing_tsit5),
            1000.0 * mean(timing_rodas5p),
            1000.0 * mean(timing_autotsit),
        ],
        "st. dev. (ms)" => [
            1000.0 * std(timing_explicit),
            1000.0 * std(timing_implicit),
            1000.0 * std(timing_tsit5),
            1000.0 * std(timing_rodas5p),
            1000.0 * std(timing_autotsit),
        ],
    )
    return timing_df
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

samples = realizations(bounds = bounds, n = 1000)
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
    HDD.ImplicitSolverPreset(
        relax = HDD.SimpleLineSearch(),
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 1.0),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Tsit5(), abstol = 1e-3, reltol = 1e-3)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Rodas5P(), abstol = 1e-3, reltol = 1e-3)),
    HDD.DiffEqSolverPreset(
        HDD.SolverConfig(alg = AutoTsit5(Rodas5P()), abstol = 1e-3, reltol = 1e-3),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = Tsit5(), abstol = 1e-6, reltol = 1e-6)),
]

results = run(parameters, solver_presets, streamflow);

scatter_plots(results)
savefig("cases/output/fuse_rsme.pdf")

df = timing_df(results)
CSV.write("cases/output/fuse_timing.csv", df)
