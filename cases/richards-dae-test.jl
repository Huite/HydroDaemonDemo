import HydroDaemonDemo as HDD
using DifferentialEquations
using Sundials
using LinearSolve

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
        kinter = HDD.ArithmeticKmean(),
    )
    return infiltration
end

forcingdf, forcing = read_forcing("data/infiltration.dat")
infiltration = create_infiltration(forcing)

solverconfig =
    HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(κ = 1e-3)), abstol = 1e-2, reltol = 1e-2)
model = HDD.diffeq_model_dae(infiltration, solverconfig, nothing)
HDD.run!(model)
plot!(model.saved[1, :])

# %%

function create_millersand()
    sand = HDD.ModifiedMualemVanGenuchten(
        a = 5.470,
        n = 4.264,
        l = 0.5,
        ks = 5.040,
        θr = 0.093,
        θs = 0.301,
        Ss = 1e-3,
        ψe = -1e-4,
    )
    #    sandspline = HDD.SplineConstitutive(sand)
    sandspline = sand
    millersand = HDD.RichardsCase(
        soil = sandspline,
        Δz = 0.0125,
        Δztotal = 10.0,
        tend = 0.18,
        dt = 0.01,
        ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
        topboundary = HDD.HeadBoundary(0.1, sandspline),
        bottomboundary = HDD.HeadBoundary(0.0, sandspline),
        forcing = nothing,
        kinter = HDD.ArithmeticKmean(),
    )
    return millersand
end

function create_millerclayloam()
    clayloam = HDD.ModifiedMualemVanGenuchten(
        a = 1.900,
        n = 1.310,
        l = 0.5,
        ks = 0.062,
        θr = 0.095,
        θs = 0.410,
        ψe = -1e-3,
        Ss = 1e-6,
    )
    clayloamspline = HDD.SplineConstitutive(clayloam)
    clayloamspline = clayloam

    millerclayloam = HDD.RichardsCase(
        soil = clayloamspline,
        Δz = 0.00625,
        Δztotal = 2.0,
        tend = 1.0,
        dt = 0.01,
        ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
        topboundary = HDD.HeadBoundary(0.1, clayloamspline),
        bottomboundary = HDD.HeadBoundary(0.0, clayloamspline),
        forcing = nothing,
        kinter = HDD.ArithmeticKmean(),
    )
    return millerclayloam
end

case = create_millerclayloam()
solverconfig =
    HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(κ = 1e-2)), abstol = 1e-6, reltol = 1e-6)
model = HDD.diffeq_model_dae(case, solverconfig, nothing)
HDD.run!(model)
plot(model.saved[:, end])

# %%

HDD.reset_and_run!(model, initial)

# %%

case = create_millerclayloam()
solverconfig =
    HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(κ = 1e-6)), abstol = 1e-2, reltol = 1e-2)
model = HDD.diffeq_model(case, solverconfig, nothing)
HDD.run!(model)

solverconfig =
    HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(κ = 1e-6)), abstol = 1e-2, reltol = 1e-2)
model2 = HDD.diffeq_model_dae(case, solverconfig, nothing)
HDD.run!(model2)
plot(model.saved[:, end])
plot!(model2.saved[:, end])
