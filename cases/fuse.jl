##
using HydroDaemonDemo
using Plots
using Sundials
using BenchmarkTools
using DifferentialEquations
##

const DAY = 24.0 * 3600.0

forcing = HydroDaemonDemo.read_forcing("cases/forcing.csv")
#forcing.precipitation ./= DAY
#forcing.evaporation ./= DAY
fuse = HydroDaemonDemo.Fuse070Parameters(
    ϕtens = 0.5,
    S1max = 0.1,
    b = 0.2,
    ku = 0.5,
    c = 10,
    v = 0.1,
    forcing = forcing,
)
initial = zeros(2) .+ 0.05
tspan = (0.0, 1000.0)

explicit_fuse = HydroDaemonDemo.ExplicitHydrologicalModel(
    fuse,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(1.0),
)
HydroDaemonDemo.run!(explicit_fuse)

plot(explicit_fuse.saved[1, :])

@btime HydroDaemonDemo.reset_and_run!(explicit_fuse, initial)

# 9.6 us

##

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverThomas(2),
    relax = HydroDaemonDemo.ScalarRelaxation(0.0),
)
implicit_fuse = HydroDaemonDemo.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(1.0),
)
HydroDaemonDemo.run!(implicit_fuse)
plot!(implicit_fuse.saved[1, :])

@btime HydroDaemonDemo.reset_and_run!(implicit_fuse, initial)

# 100 us


##

##

solverconfig = HydroDaemonDemo.SolverConfig(alg = Tsit5())
diffeq_fuse =
    HydroDaemonDemo.DiffEqHydrologicalModel(fuse, initial, tspan, nothing, solverconfig)
HydroDaemonDemo.run!(diffeq_fuse)

plot!(diffeq_fuse.saved[1, :])


fuse550 = HydroDaemonDemo.Fuse550Parameters(
    ϕtens = 0.5,
    S1max = 0.2,
    S2max = 0.2,
    r1 = 1.0,
    b = 0.2,
    ku = 0.01,
    ki = 0.01,
    ks = 0.05,
    c = 1.0,
    n = 2.0,
    Acmax = 0.5,
    forcing = forcing,
)
initial = zeros(2) .+ 0.05
tspan = (0.0, 1000.0)

explicit_fuse = HydroDaemonDemo.ExplicitHydrologicalModel(
    fuse550,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(1.0),
)
HydroDaemonDemo.run!(explicit_fuse)

plot(explicit_fuse.saved[1, :])
plot!(explicit_fuse.saved[2, :])
