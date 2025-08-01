using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
using Plots


function implicit_model(parameters, ψ0, tspan, Δt, saveat)
    n = length(ψ0)
    solver = HydroDaemonDemo.NewtonSolver(
        HydroDaemonDemo.LinearSolverThomas(n),
        relax = HydroDaemonDemo.ScalarRelaxation(0.5),
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


loamproperties = HydroDaemonDemo.MualemVanGenuchten(
    a = 3.600,
    n = 1.560,
    l = 0.5,
    ks = 0.250,
    θr = 0.078,
    θs = 0.430,
    Ss = 1e-6,
)
loamproperties = HydroDaemonDemo.SplineConstitutive(loamproperties)

Δz = 0.0125
Δztotal = 5.0
Δt = 0.001
tend = 2.25
n = Int(Δztotal / Δz)
constitutive = fill(loamproperties, n)

parameters = HydroDaemonDemo.RichardsParameters(
    constitutive,
    Δz,
    HydroDaemonDemo.MeteorologicalForcing([0.0], [0.0], [0.0]),
    HydroDaemonDemo.HeadBoundary(0.0, constitutive[1]),
    HydroDaemonDemo.HeadBoundary(0.1, constitutive[end]),
)

ψ0 = -collect(Δz:Δz:Δztotal)
tspan = (0.0, tend)
saveat = collect(0.0:Δt:tend)

implicit_richards = implicit_model(parameters, ψ0, tspan, Δt, saveat)
HydroDaemonDemo.run!(implicit_richards)
plot(implicit_richards.saved[:, end], permute = (:y, :x))

@btime HydroDaemonDemo.reset_and_run!(implicit_richards, ψ0)

function diffeq_model(parameters, ψ0, tspan, Δt, saveat)
    solverconfig = HydroDaemonDemo.SolverConfig(
        alg = QBDF(autodiff = true, nlsolve = NLNewton(relax = 0.5)),
        abstol = 1e-6,
        reltol = 1e-6,
        maxiters = 50000,
        detect_sparsity = false,
    )
    diffeq_richards =
        HydroDaemonDemo.DiffEqHydrologicalModel(parameters, ψ0, tspan, saveat, solverconfig)
    return diffeq_richards
end
diffeq_richards = diffeq_model(parameters, ψ0, tspan, Δt, saveat)
HydroDaemonDemo.run!(diffeq_richards)
plot!(diffeq_richards.saved[:, end], permute = (:y, :x))

#@btime HydroDaemonDemo.reset_and_run!(diffeq_richards, ψ0)


function diffeq_model_dae(parameters, ψ0, tspan, Δt, saveat)
    dae_parameters = HydroDaemonDemo.RichardsParametersDAE(parameters)
    θ0 = [
        HydroDaemonDemo.moisture_content(ψ, c) for
        (ψ, c) in zip(ψ0, parameters.constitutive)
    ]
    u0 = [ψ0; θ0]
    solverconfig = HydroDaemonDemo.SolverConfig(
        alg = QBDF(autodiff = true, nlsolve = NLNewton(relax = 0.5)),
        abstol = 1e-6,
        reltol = 1e-6,
        maxiters = 50000,
        detect_sparsity = false,
    )
    diffeq_richards_dae = HydroDaemonDemo.DiffEqHydrologicalModel(
        dae_parameters,
        u0,
        tspan,
        saveat,
        solverconfig,
    )
    return diffeq_richards_dae
end


diffeq_richards_dae = diffeq_model_dae(parameters, ψ0, tspan, Δt, saveat)
HydroDaemonDemo.run!(diffeq_richards_dae)
plot!(diffeq_richards_dae.saved[1:parameters.n, end], permute = (:y, :x))


@btime HydroDaemonDemo.reset_and_run!(diffeq_richards_dae, ψ0)
