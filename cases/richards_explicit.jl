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
    Δz,
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(-61.5, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(-21.0, constitutive[end]),
)
tspan = (0.0, 5.0)
saveat = collect(0.0:1.0:360.0)
explicit_richards = HydroDaemonDemo.ExplicitHydrologicalModel(
    parameters,
    initial,
    tspan,
    saveat,
    #    HydroDaemonDemo.FixedTimeStepper(0.1),
    HydroDaemonDemo.CFLTimeStepper(0.1, 0.5, 1e-6),
)
HydroDaemonDemo.run!(explicit_richards)

##

using Plots
plot(explicit_richards.saved[:, end])

##

cell = constitutive[1]
HydroDaemonDemo.conductivity(100.5, cell)
