using HydroDaemonDemo
using DifferentialEquations
using BenchmarkTools
using Plots

const Float = Float64

@kwdef struct Case{T}
    parameters::T
    Δz::Float
    Δztotal::Float
    Δt::Float
    tend::Float
end

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
        HydroDaemonDemo.AdaptiveTimeStepper(Δt, Δtmin = 1e-8),
    )
    return implicit_richards
end

function diffeq_model(parameters, ψ0, tspan, Δt, saveat)
    solverconfig = HydroDaemonDemo.SolverConfig(
        dt = Δt,
        dtmin = 1e-8,
        dtmax = 1.0,
        alg = ImplicitEuler(autodiff = true, nlsolve = NLNewton()),
        adaptive = true,
        force_dtmin = false,
        abstol = 1e-5,
        reltol = 1e-5,
        maxiters = 1000,
        detect_sparsity = false,
    )
    diffeq_richards =
        HydroDaemonDemo.DiffEqHydrologicalModel(parameters, ψ0, tspan, saveat, solverconfig)
    return diffeq_richards
end

function diffeq_model_dae(parameters, ψ0, tspan, Δt, saveat)
    dae_parameters = HydroDaemonDemo.RichardsParametersDAE(parameters)
    u0 = [ψ0; zero(ψ0)]
    solverconfig = HydroDaemonDemo.SolverConfig(
        dt = Δt,
        dtmin = 1e-8,
        dtmax = 1.0,
        alg = QNDF(autodiff = true, nlsolve = NLNewton()),
        adaptive = true,
        force_dtmin = false,
        abstol = 1e-5,
        reltol = 1e-5,
        maxiters = 1000,
        detect_sparsity = false,
    )
    diffeq_richards = HydroDaemonDemo.DiffEqHydrologicalModel(
        dae_parameters,
        u0,
        tspan,
        saveat,
        solverconfig,
    )
    return diffeq_richards
end

function run_case(case)
    (; parameters, Δz, Δztotal, Δt, tend) = case
    n = Int(Δztotal / Δz)
    constitutive = fill(parameters, n)
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
    diffeq_richards = diffeq_model(parameters, ψ0, tspan, Δt, saveat)
    diffeq_richards_dae = diffeq_model_dae(parameters, ψ0, tspan, Δt, saveat)
    HydroDaemonDemo.run!(implicit_richards)
#    HydroDaemonDemo.run!(diffeq_richards)
#    HydroDaemonDemo.run!(diffeq_richards_dae)

    #result_implicit = @btime HydroDaemonDemo.reset_and_run!($implicit_richards, $ψ0)
    #result_diffeq = @btime HydroDaemonDemo.reset_and_run!($diffeq_richards, $ψ0)
    ##result_dae = @btime HydroDaemonDemo.reset_and_run!($diffeq_richards_dae, $ψ0)
    return implicit_richards.saved[:, end], diffeq_richards.saved[:, end]
end

ψmin = -1e4
sandproperties = HydroDaemonDemo.MualemVanGenuchten(
    a = 5.470,
    n = 4.264,
    l = 0.5,
    ks = 5.040,
    θr = 0.093,
    θs = 0.301,
)
loamproperties = HydroDaemonDemo.MualemVanGenuchten(
    a = 3.600,
    n = 1.560,
    l = 0.5,
    ks = 0.250,
    θr = 0.078,
    θs = 0.430,
)
clayloamproperties = HydroDaemonDemo.MualemVanGenuchten(
    a = 1.900,
    n = 1.310,
    l = 0.5,
    ks = 0.062,
    θr = 0.095,
    θs = 0.410,
)

sand = Case(
    parameters = HydroDaemonDemo.HermiteSplineConstitutive(sandproperties, ψmin)[1],
    Δz = 0.0125,
    Δztotal = 10.0,
    Δt = 0.001,
    tend = 0.18,
)
loam = Case(
    parameters = HydroDaemonDemo.HermiteSplineConstitutive(loamproperties, ψmin)[1],
    Δz = 0.0125,
    Δztotal = 5.0,
    Δt = 0.001,
    tend = 2.25,
)
clay_loam = Case(
    parameters = HydroDaemonDemo.HermiteSplineConstitutive(clayloamproperties, ψmin)[1],
    Δz = 0.0125,
    Δztotal = 2.0,
    Δt = 0.001,
    tend = 1.0,
)

ψsand_implicit, ψsand_diffeq = run_case(sand)
loam_implicit = run_case(loam)
clayloam_implicit = run_case(clay_loam)

plot(ψsand_implicit, permute=(:y, :x))
plot(ψsand_diffeq, permute=(:y, :x))
plot!(ψloam, permute=(:y, :x))
plot!(ψclay_loam, permute=(:y, :x))


(; parameters, Δz, Δztotal, Δt, tend) = loam
n = Int(Δztotal / Δz)
constitutive = fill(parameters, n)
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
@btime HydroDaemonDemo.reset_and_run!(implicit_richards, ψ0)