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
    return HDD.MeteorologicalForcing(df.time, P, zero(P))
end

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
spline = HDD.SplineConstitutive(siltloam)
forcing = read_forcing("data/infiltration.dat")

infiltration = HDD.RichardsCase(
    soil = siltloam,
    Δz = 0.1,
    Δztotal = 1.5,
    tend = forcing.t[end] + 1.0,
    ψ0 = HDD.InitialConstant(-3.59),
    bottomboundary = HDD.FreeDrainage(),
    topboundary = nothing,
    forcing = forcing,
)
saveat = collect(0.0:1.0:infiltration.tspan[2])

implicit_solver = HDD.NewtonSolver(
    HDD.LinearSolverThomas(infiltration.parameters.n),
    relax = HDD.ScalarRelaxation(0.0),
    abstol = 1e-6,
    reltol = 1e-6,
)
implicit_result = HDD.benchmark!(
    HDD.implicit_model(infiltration, implicit_solver, HDD.AdaptiveTimeStepper(1.0), saveat),
    infiltration,
)  # 143 ms

wb = implicit_result.waterbalance
ΔS = diff(wb.storage)
qb = diff(wb.qbot)
qt = diff(wb.qtop)
error = qt + qb - ΔS

scatter(qt, error)

qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = QNDF(), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)  # 813 ms, dae: 1.16 s

wb = qndf_result.waterbalance
ΔS = diff(wb.storage)
qb = diff(wb.qbot)
qt = diff(wb.qtop)
error = qt + qb - ΔS

scatter(qt, error)


qndf_result_dae = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = QNDF(), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)  # 813 ms, dae: 1.16 s


wb = qndf_result_dae.waterbalance
ΔS = diff(wb.storage)
qb = diff(wb.qbot)
qt = diff(wb.qtop)
error = qt + qb - ΔS

scatter!(qt, error)

model = qndf_result.model

n = model.integrator.p.parameters.n
S = [HDD.storage(ψ, model.integrator.p.parameters) for ψ in eachcol(model.saved[1:n, :])]
total_storage = sum.(S)
qbot_cumulative = model.saved[end-1, :]
qtop_cumulative = model.saved[end, :]

qb = qbot_cumulative[end]
qt = qtop_cumulative[end]
dS = total_storage[end] - total_storage[1]
qb + qt - dS


qndf_result_dae = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = QNDF(), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)  # 813 ms, dae: 1.16 s

qndf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = QNDF(autodiff = false), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)  #  865 ms, dae: 1.17 s

qbdf_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = QBDF(), maxiters = 100_000, reltol = 1e-3),
        saveat,
    ),
    infiltration,
)  # 853 ms, dae: 1.15 s
# reltol 1e-3, 330 ms, dae: 700 ms

cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    infiltration,
)  # 837 ms

euler_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 200_000),
        saveat,
    ),
    infiltration,
)  # Excess of maxiter 500 000

euler_custom_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(
            alg = ImplicitEuler(),
            maxiters = 100_000,
            controller = HDD.CustomController(dtmin = 1e-9),
        ),
        saveat,
    ),
    infiltration,
)  # 2.5 s

euler_dae_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(
            alg = ImplicitEuler(),
            maxiters = 100_000,
            controller = HDD.CustomController(dtmin = 1e-9),
            detect_sparsity = true,
        ),
        saveat,
    ),
    infiltration,
)  # 412 ms

storage = [
    HDD.storage(ψ, infiltration.parameters) for
    ψ in eachcol(euler_dae_result.model.saved[1:15, :])
]
total_storage = sum.(storage)
plot!(date_range, total_storage[1554:1554+365], label = "ImplicitEulerDAE(custom)")


euler_dae_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(
            alg = ImplicitEuler(autodiff = false),
            maxiters = 100_000,
            controller = HDD.CustomController(dtmin = 1e-9),
        ),
        saveat,
    ),
    infiltration,
)  # 461 ms

fbdf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = FBDF(), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)  # 1.1 s, dae: 1.5 s


start_date = Date("2004-01-01")
end_date = Date("2004-12-31")
date_range = collect(start_date:Day(1):end_date)
#plot(date_range, total_storage[1554:1554+365], xlabel="Date", ylabel="Storage (m)", label="Implicit")
date_range = df.Column1

storage =
    [HDD.storage(ψ, infiltration.parameters) for ψ in eachcol(implicit_result.model.saved)]
total_storage = sum.(storage)
plot(
    date_range,
    total_storage[1:end-1],
    xlabel = "Date",
    ylabel = "Storage (m)",
    label = "Implicit",
)


storage =
    [HDD.storage(ψ, infiltration.parameters) for ψ in eachcol(cvode_result.model.saved)]
total_storage = sum.(storage)
plot!(
    date_range,
    total_storage[1:end-1],
    xlabel = "Date",
    ylabel = "Storage (m)",
    label = "CVODE_BDF",
)

storage =
    [HDD.storage(ψ, infiltration.parameters) for ψ in eachcol(qndf_result.model.saved)]
total_storage = sum.(storage)
plot!(date_range, total_storage[1:end-1], label = "QNDF")

storage = [
    HDD.storage(ψ, infiltration.parameters) for
    ψ in eachcol(qndf_result_dae.model.saved[1:15, :])
]
total_storage = sum.(storage)
plot!(date_range, total_storage[1:end-1], label = "QNDF-DAE")

storage = [
    HDD.storage(ψ, infiltration.parameters) for
    ψ in eachcol(euler_custom_result.model.saved)
]
total_storage = sum.(storage)
plot!(date_range, total_storage[1:end-1], label = "ImplicitEuler(custom)")

storage = [
    HDD.storage(ψ, infiltration.parameters) for
    ψ in eachcol(euler_dae_result.model.saved[1:15, :])
]
total_storage = sum.(storage)
plot!(date_range, total_storage[1:end-1], label = "ImplicitEuler-DAE(custom)")
