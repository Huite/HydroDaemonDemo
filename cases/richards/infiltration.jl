import HydroDaemonDemo as HDD
using DifferentialEquations
using Sundials

using Plots
using CSV
using DataFrames
using Dates


function read_forcing(path)
    df = CSV.read(path, DataFrame)
    rename!(df, "Column1" => "Date")
    df.time = Dates.value.(df.Date - df.Date[1]) / 1.0
    P = df[!, "Precipitation (mm/d)"] / 1000.0
    return df, HDD.MeteorologicalForcing(df.time, P, zero(P))
end

function create_infiltration(forcing)
    # Note: units are meters and days!
    siltloam = HDD.ModifiedMualemVanGenuchten(
        a = 0.423,
        n = 2.06,
        l = 0.5,
        ks = 0.0496,  # m/d
        θr = 0.131,
        θs = 0.396,
        ψe = -1e-3,
        Ss = 1e-6,
    )
    siltloamspline = HDD.SplineConstitutive(siltloam)
    infiltration = HDD.RichardsCase(
        soil = siltloamspline,
        Δz = 0.1,
        Δztotal = 1.5,
        tend = forcing.t[end] + 1.0,
        dt = 1.0,
        ψ0 = HDD.InitialConstant(-3.59),
        bottomboundary = HDD.FreeDrainage(),
        topboundary = nothing,
        forcing = forcing,
    )
    return infiltration
end

function run(case, solver_presets)
    rows = []
    results = []
    for preset in solver_presets
        println("Benchmarking infiltration, $(HDD.name(preset))")
        result = HDD.benchmark!(case, preset)
        push!(results, result)
        push!(
            rows,
            (
                solver = HDD.name(preset),
                #time = minimum(result.trial).time / 1e9,
                time = result.time,
                mass_bias = result.mass_bias,
                mass_rmse = result.mass_rsme,
            ),
        )
    end
    df = DataFrame(rows)
    return df, results
end

forcingdf, forcing = read_forcing("data/infiltration.dat")
infiltration = create_infiltration(forcing)
solver_presets = (
    HDD.ImplicitSolverPreset(
        relax = HDD.ScalarRelaxation(0.0),
        abstol = 1e-6,
        reltol = 1e-6,
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 1.0),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1))),
)

df, results = run(infiltration, solver_presets)
CSV.write("cases/richards/infiltration.csv", df)

function plot_storage(forcingdf, case, solver_presets, results)
    start_date = Date("2004-01-01")
    end_date = Date("2004-12-31")
    date_range = collect(start_date:Day(1):end_date)
    start_index = findfirst(x -> x == start_date, forcingdf.Date)
    end_index = findfirst(x -> x == end_date, forcingdf.Date)

    plot()
    for (preset, result) in zip(solver_presets, results)
        plot!(
            date_range,
            result.waterbalance.storage[start_index:end_index],
            label = HDD.name(preset),
            xlabel = "Date",
            ylabel = "Storage (m)",
        )
    end
    display(current())
    return
end


function plot_drainage(forcingdf, case, solver_presets, results)
    start_date = Date("2004-01-01")
    end_date = Date("2004-12-31")
    date_range = collect(start_date:Day(1):end_date)
    start_index = findfirst(x -> x == start_date, forcingdf.Date)
    end_index = findfirst(x -> x == end_date, forcingdf.Date)

    plot()
    for (preset, result) in zip(solver_presets, results)
        plot!(
            date_range[2:end],
            -1.0 .* diff(result.waterbalance.qbot[start_index:end_index]),
            label = HDD.name(preset),
            xlabel = "Date",
            ylabel = "Drainage (m/d)",
        )
    end
    display(current())
    return
end


plot_storage(forcingdf, infiltration, solver_presets, results)
savefig("cases/richards/infiltration-storage.png")
savefig("cases/richards/infiltration-storage.svg")


plot_drainage(forcingdf, infiltration, solver_presets, results[1:(end-2)])
savefig("cases/richards/infiltration-drainage.png")
savefig("cases/richards/infiltration-drainage.svg")



plot_drainage(forcingdf, infiltration, solver_presets, results)
savefig("cases/richards/infiltration-drainage-custom.png")
savefig("cases/richards/infiltration-drainage-custom.svg")
