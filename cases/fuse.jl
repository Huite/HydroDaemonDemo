##
using HydroDaemonDemo
using Plots
using BenchmarkTools
using DifferentialEquations
##

const DAY = 24.0 * 3600.0

forcing = HydroDaemonDemo.read_forcing("cases/forcing.csv")
forcing.precipitation .*= 0.1
fuse = HydroDaemonDemo.Fuse070Parameters(
    ϕtens = 0.5,
    S1max = 100.0,
    ku = 0.5 / DAY,
    c = 10,
    v = 0.1,
    m = 0.6 * DAY,
    b = 0.2,
    forcing = forcing,
)
initial = zeros(2) .+ 5.0
tspan = (0.0, 100.0 * DAY)

explicit_fuse = HydroDaemonDemo.ExplicitHydrologicalModel(
    fuse,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(0.1 * DAY),
)
HydroDaemonDemo.run!(explicit_fuse)

plot(explicit_fuse.saved[1, :])

##

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverLU(2),
    relax = HydroDaemonDemo.ScalarRelaxation(0.0),
)
implicit_fuse = HydroDaemonDemo.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HydroDaemonDemo.AdaptiveTimeStepper(0.1 * DAY),
)
HydroDaemonDemo.run!(implicit_fuse)

##
plot!(implicit_fuse.saved[1, :])

##

solverconfig = HydroDaemonDemo.SolverConfig(alg = Tsit5())
diffeq_fuse =
    HydroDaemonDemo.DiffEqHydrologicalModel(fuse, initial, tspan, nothing, solverconfig)
HydroDaemonDemo.run!(diffeq_fuse)

plot(diffeq_fuse.saved[1, :])


##

plot(explicit_fuse.saved[1, :])
#plot!(implicit_fuse.saved[1, :])
plot!(out[1, :])
