##
using HydroDaemonDemo
using Plots
using BenchmarkTools
using DifferentialEquations
##

const DAY = 24.0 * 3600.0

area = 100.0
n = 5
a = 1e-8
b = 1.5
forcing = HydroDaemonDemo.read_forcing("cases/forcing.csv")
cascade = HydroDaemonDemo.BucketCascade(fill(area, n), fill(a, n), fill(b, n), forcing)
initial = zeros(5)
tspan = (0.0, 100.0 * DAY)

explicit_reservoirs = HydroDaemonDemo.ExplicitHydrologicalModel(
    cascade,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(0.01 * DAY),
)
HydroDaemonDemo.run!(explicit_reservoirs)

##

solver = HydroDaemonDemo.NewtonSolver(
    HydroDaemonDemo.LinearSolverThomas(n),
    relax = HydroDaemonDemo.CubicLineSearch(
        a0 = 1.0,
        c = 1e-3,
        maxiter = 5,
        low = 1e-3,
        high = 1.0,
    ),
)
implicit_reservoirs = HydroDaemonDemo.ImplicitHydrologicalModel(
    cascade,
    initial,
    solver,
    tspan,
    nothing,
    HydroDaemonDemo.AdaptiveTimeStepper(1.0 * DAY),
)
HydroDaemonDemo.run!(implicit_reservoirs)

##

solverconfig = HydroDaemonDemo.SolverConfig(
    dt = 1.0 * DAY,
    dtmin = 1.0,
    dtmax = 1.0 * DAY;
    alg = ImplicitEuler(),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-3,
    reltol = 1e-3,
    maxiters = 10000,
)

diffeq_reservoirs =
    HydroDaemonDemo.DiffEqHydrologicalModel(cascade, initial, tspan, nothing, solverconfig)
HydroDaemonDemo.run!(diffeq_reservoirs)

plot(implicit_reservoirs.saved[1, :])
plot!(explicit_reservoirs.saved[1, :])
plot!(diffeq_reservoirs.saved[1, :])

##

@btime HydroDaemonDemo.reset_and_run!(explicit_reservoirs, 0.0);  # 10 us at dt=1.0, 866 us at dt=0.01
@btime HydroDaemonDemo.reset_and_run!(implicit_reservoirs, 0.0);  # 11.9 ms
@btime HydroDaemonDemo.reset_and_run!(diffeq_reservoirs, 0.0);   # 4.8 ms  # EulerImplicit()
