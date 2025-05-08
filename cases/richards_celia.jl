##
using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
##

n = 40
Δz = 1.0  # cm

# Celia case

constitutive = fill(
    HydroDaemonDemo.Haverkamp(
        α = 1.611e6,
        β = 3.96,
        γ = 4.74,
        A = 1.175e6,
        ks = 0.00944,
        θs = 0.287,
        θr = 0.075,
    ),
    n,
)
initial = fill(-61.5, n)
parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    fill(Δz, n),
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
    HydroDaemonDemo.FixedTimeStepper(Δt),
)
HydroDaemonDemo.run!(implicit_richards)

# O relax
# 3.7 ms

# Simple line search
# 5.9 ms

# CubicLineSearch
# 5.9 ms

##

using LineSearches: BackTracking

solverconfig = HydroDaemonDemo.SolverConfig(
    dt = 1.0,
    dtmin = 1e-6,
    dtmax = 1.0,
    alg = ImplicitEuler(autodiff = false, nlsolve = NLNewton()),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-5,
    reltol = 1e-5,
    maxiters = 10000,
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

@btime HydroDaemonDemo.reset_and_run!(diffeq_richards, -61.5)

# Relax 0.5
# QNDF: 4.83 ms

# Relax 0.0
# ImplicitEuler: 49.8 ms
# Rosenbrock23: 67.98 ms
# QNDF: 3.96 ms

# Tsit5: 7.86 ms

##

using Plots
plot(implicit_richards.saved[:, end])
plot!(diffeq_richards.saved[:, end])
