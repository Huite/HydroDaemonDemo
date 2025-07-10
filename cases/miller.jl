using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
using Plots

sand = HydroDaemonDemo.MualemVanGenuchten(
    a = 5.470,
    n = 4.264,
    m = 1 - 1 / 4.264,
    l = 0.5,
    ks = 5.040,  # m/d
    θr = 0.093,
    θs = 0.301,
)
loam = HydroDaemonDemo.MualemVanGenuchten(
    a = 3.600,
    n = 1.560,
    m = 1 - 1 / 1.56,
    l = 0.5,
    ks = 0.250,  # m/d
    θr = 0.078,
    θs = 0.430,
)
clay_loam = HydroDaemonDemo.MualemVanGenuchten(
    a = 1.900,
    n = 1.310,
    m = 1 - 1 / 1.310,
    l = 0.5,
    ks = 0.062,  # m/d
    θr = 0.095,
    θs = 0.410,
)


Δz = 0.0125
Ss = 1e-6
ztotal = 5.0
n = Int(5.0 / Δz)
constitutive = fill(loam, n)

parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    Δz,
    Ss,
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(0.0, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(0.1, constitutive[end]),
)
initial = -collect(Δz:Δz:ztotal)
tend = 0.18
tspan = (0.0, tend)
Δt = 0.001
saveat = collect(0.0:Δt:tend)

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverThomas(n),
    relax = HydroDaemonDemo.ScalarRelaxation(0.5),
)
implicit_richards = HydroDaemonDemo.ImplicitHydrologicalModel(
    parameters,
    initial,
    solver,
    tspan,
    saveat,
    HydroDaemonDemo.AdaptiveTimeStepper(Δt, Δtmin = 1e-6),
)
HydroDaemonDemo.run!(implicit_richards)
plot(implicit_richards.saved[:, 20])
