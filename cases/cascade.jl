##
using HydroDaemonDemo
using CSV
using DataFrames
using Dates
using DifferentialEquations
using Plots
using BenchmarkTools
##

const DAY = 24.0 * 3600.0

area = 100.0
n = 5
a = 1e-8
b = 1.5

df = CSV.read("cases/forcing.csv", DataFrame)
df.time = Dates.toms.(df.Date - df.Date[1]) / 1000.0
forcing =
    HydroDaemonDemo.MeteorologicalForcing(df.time, df.P / 1000.0, df.ET / 1000.0)
cascade =
    HydroDaemonDemo.bucket_cascade_analytic(fill(area, n), fill(a, n), fill(b, n), forcing)
initial = zeros(5)
tspan = (0.0, 100.0 * DAY)

##

explicit_reservoirs = HydroDaemonDemo.ExplicitHydrologicalModel(
    cascade,
    initial,
    tspan,
    nothing,
    HydroDaemonDemo.FixedTimeStepper(1.0 * DAY),
)
HydroDaemonDemo.run!(explicit_reservoirs)

##

solver = HydroDaemonDemo.NewtonSolver(linearsolver = HydroDaemonDemo.LinearSolverLU(n))
implicit_reservoirs = HydroDaemonDemo.ImplicitHydrologicalModel(
    cascade,
    initial,
    solver,
    tspan,
    nothing,
    HydroDaemonDemo.AdaptiveTimeStepper(1.0 * DAY),
)

##

HydroDaemonDemo.run!(implicit_reservoirs)

plot(implicit_reservoirs.saved[1, :])
plot!(explicit_reservoirs.saved[1, :])


##

solverconfig =
    HydroDaemonDemo.SolverConfig(Tsit5(), true, 1e-3, 1e-6, 1.0, false, 1e-6, 1e-6, 250)
diffeq_reservoirs = HydroDaemonDemo.DiffEqHydrologicalModel(
    HydroDaemonDemo.reservoir_rhs!,
    cascade,
    initial,
    tspan,
    nothing,
    solverconfig,
)
HydroDaemonDemo.run!(diffeq_reservoirs)


## Check for allocations

function reset_and_run!(model)
    state = primary(model.state)
    state .= 0.0
    HydroDaemonDemo.run!(explicit_reservoirs)
end
@btime reset_and_run!(explicit_reservoirs)
