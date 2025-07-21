using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
using Plots

const Float = Float64

function implicit_model(parameters, ψ0, tspan, Δt, saveat)
    n = length(ψ0)
    solver = HydroDaemonDemo.NewtonSolver(
        HydroDaemonDemo.LinearSolverThomas(n),
        relax = HydroDaemonDemo.ScalarRelaxation(0.0),
        tolerance = 1e-6,
    )
    implicit_richards = HydroDaemonDemo.ImplicitHydrologicalModel(
        parameters,
        ψ0,
        solver,
        tspan,
        saveat,
        HydroDaemonDemo.AdaptiveTimeStepper(Δt, Δtmin = 1e-5),
    )
    return implicit_richards
end


sandproperties = HydroDaemonDemo.MualemVanGenuchten(
    a = 5.470,
    n = 4.264,
    l = 0.5,
    ks = 5.040,
    θr = 0.093,
    θs = 0.301,
)
#sandproperties, error = HydroDaemonDemo.SplineConstitutive(
#    sandproperties,
#    relative_error=1e-3,
#    ψmin=-1e2,
#    ψe=0.0,
#    nknots=50,
#    iter=5,
#)

Δz = 0.0125
Δztotal = 10.0
Δt = 0.001
tend = 0.18
n = Int(Δztotal / Δz)
constitutive = fill(sandproperties, n)
Ss = 1e-6

parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    Δz,
    Ss,
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(0.0, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(0.1, constitutive[end]),
)

ψ0 = -collect(Δz:Δz:Δztotal)
tspan = (0.0, tend)
saveat = collect(0.0:Δt:tend)

implicit_richards = implicit_model(parameters, ψ0, tspan, Δt, saveat)
HydroDaemonDemo.run!(implicit_richards)
plot(implicit_richards.saved[:, end], permute=(:y, :x))


function diffeq_model(parameters, ψ0, tspan, Δt, saveat)
    solverconfig = HydroDaemonDemo.SolverConfig(
        dt = 1e-9,
        dtmin = 1e-10,
        dtmax = 1.0,
        alg = ImplicitEuler(autodiff = false, nlsolve = NLNewton()),
        adaptive = true,
        force_dtmin = false,
        abstol = 1e-6,
        reltol = 1e-6,
        maxiters = 1000,
        detect_sparsity = false,
    )
    diffeq_richards =
        HydroDaemonDemo.DiffEqHydrologicalModel(parameters, ψ0, tspan, saveat, solverconfig)
    return diffeq_richards
end
diffeq_richards = diffeq_model(parameters, ψ0, tspan, Δt, saveat)
HydroDaemonDemo.run!(diffeq_richards)
plot!(diffeq_richards.saved[:, end], permute=(:y, :x))
