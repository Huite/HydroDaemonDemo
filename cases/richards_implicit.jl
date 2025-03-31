##
using HydroDaemonDemo
##

n = 40
Δz = 1.0  # cm

# Celia case

constitutive = [
    HydroDaemonDemo.Haverkamp(
        α = 1.611e6,
        β = 3.96,
        γ = 4.74,
        A = 1.175e6,
        ks = 0.00944,
        θs = 0.287,
        θr = 0.075,
    ) for _ = 1:n
]
initial = fill(-61.5, n)
parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    fill(Δz, n),
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(-61.5, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(-20.5, constitutive[end]),
    #    HydroDaemonDemo.FreeDrainage(),
    #    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.1], [0.0]),
)
tend = 360.0
tspan = (0.0, tend)

Δt = 360.0
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
state = implicit_richards.state

out = implicit_richards.saved

using Plots
#plot(out[:, 1])
plot(out[:, end])
