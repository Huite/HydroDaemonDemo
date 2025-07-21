##
using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
using Plots
##

n = 40
Δz = 1.0  # cm
Ss = 1e-6

# Celia case

soilproperties = HydroDaemonDemo.Haverkamp(
    a = 1.611e6,
    β = 3.96,
    y = 4.74,
    A = 1.175e6,
    ks = 0.00944,  # m/s
    θs = 0.287,
    θr = 0.075,
)
constitutive = fill(soilproperties, n)
initial = fill(-61.5, n)
parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    Δz,
    Ss,
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(-61.5, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(-20.5, constitutive[end]),
)
tend = 360.0
tspan = (0.0, tend)

Δt = 1.0
saveat = collect(0.0:Δt:tend)

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverThomas(n),
    relax = HydroDaemonDemo.ScalarRelaxation(0.0),
)
implicit_richards = HydroDaemonDemo.ImplicitHydrologicalModel(
    parameters,
    initial,
    solver,
    tspan,
    saveat,
    HydroDaemonDemo.AdaptiveTimeStepper(Δt),
)
HydroDaemonDemo.run!(implicit_richards)

solverconfig = HydroDaemonDemo.SolverConfig(
    dt = 1.0,
    dtmin = 1e-6,
    dtmax = 1.0,
    alg = QNDF(autodiff = true, nlsolve = NLNewton()),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-5,
    reltol = 1e-5,
    maxiters = 10000,
    detect_sparsity = false,
)

diffeq_richards = HydroDaemonDemo.DiffEqHydrologicalModel(
    parameters,
    initial,
    tspan,
    saveat,
    solverconfig,
)
HydroDaemonDemo.run!(diffeq_richards)

HydroDaemonDemo.reset_and_run!(diffeq_richards, -61.5)

parameters_dae = HydroDaemonDemo.RichardsParametersDAE(parameters)
solverconfig = HydroDaemonDemo.SolverConfig(
    dt = 1.0,
    dtmin = 1e-6,
    dtmax = 1.0,
    alg = QNDF(autodiff = true, nlsolve = NLNewton()),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-5,
    reltol = 1e-5,
    maxiters = 10000,
    detect_sparsity = false,
)
diffeq_richards_dae = HydroDaemonDemo.DiffEqHydrologicalModel(
    parameters_dae,
    [initial; zero(initial)],
    tspan,
    saveat,
    solverconfig,
)
HydroDaemonDemo.run!(diffeq_richards_dae)
HydroDaemonDemo.reset_and_run!(diffeq_richards_dae, -61.5)

plot(implicit_richards.saved[:, end])
plot!(diffeq_richards.saved[:, end])
plot!(diffeq_richards_dae.saved[1:n, end])

@btime HydroDaemonDemo.reset_and_run!(implicit_richards, -61.5)
@btime HydroDaemonDemo.reset_and_run!(diffeq_richards, -61.5)
@btime HydroDaemonDemo.reset_and_run!(diffeq_richards_dae, -61.5)

plot(implicit_richards.saved[:, end])
plot!(diffeq_richards.saved[:, end])
plot!(diffeq_richards_dae.saved[1:n, end])