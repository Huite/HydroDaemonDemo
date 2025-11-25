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
                time = minimum(result.trial).time / 1e9,
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
    HDD.ImplicitNewtonSolverPreset(
        relax = HDD.ScalarRelaxation(0.0),
        abstol = 1e-8,
        reltol = 1e-8,
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 1.0),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1))),
)

df, results = run(infiltration, solver_presets)
CSV.write("cases/output/infiltration.csv", df)

dates = copy(forcingdf.Date)
push!(dates, dates[end] + Day(1))
storagedf = DataFrame(date = dates)
for (preset, result) in zip(solver_presets, results)
    storagedf[!, HDD.name(preset)] = result.waterbalance.storage
end

CSV.write("cases/output/infiltration-storage.csv", storagedf)

drainagedf = DataFrame(date = dates[2:end])
for (preset, result) in zip(solver_presets, results)
    drainagedf[!, HDD.name(preset)] = -1.0 .* diff(result.waterbalance.qbot)
end

CSV.write("cases/output/infiltration-drainage.csv", drainagedf)
