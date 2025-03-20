##
using Revise
using HydroDaemonDemo
using Plots
using BenchmarkTools
using DifferentialEquations
##

const DAY = 24.0 * 3600.0

forcing = HydroDaemonDemo.read_forcing("cases/forcing.csv")
forcing.evaporation .*= 0.0
fuse = HydroDaemonDemo.Fuse070Parameters(forcing)  # parameters left at default value
initial = zeros(2) .+ 5.0
tspan = (0.0, 100.0 * DAY)

explicit_fuse = HydroDaemonDemo.ExplicitHydrologicalModel(
    fuse,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(1.0 * DAY),
)
HydroDaemonDemo.run!(explicit_fuse)

plot(explicit_fuse.saved[1, :])

##

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverLU(2),
    relax = HydroDaemonDemo.ScalarRelaxation(0.5),
)
implicit_fuse = HydroDaemonDemo.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(0.1 * DAY),
)
HydroDaemonDemo.run!(implicit_fuse)

##
plot!(implicit_fuse.saved[1, :])

##

solverconfig = HydroDaemonDemo.SolverConfig(
    1.0 * DAY,
    1e-9 * DAY,
    1.0 * DAY;
    alg = ImplicitEuler(),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-3,
    reltol = 1e-3,
    maxiters = 10000,
)
diffeq_fuse =
    HydroDaemonDemo.DiffEqHydrologicalModel(fuse, initial, tspan, nothing, solverconfig)
out = HydroDaemonDemo.run!(diffeq_fuse)



##

plot(explicit_fuse.saved[1, :])
#plot!(implicit_fuse.saved[1, :])
plot!(out[1, :])
