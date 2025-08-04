import HydroDaemonDemo as HDD
using DifferentialEquations
using Sundials


sand = HDD.ModifiedMualemVanGenuchten(
    a = 5.470,
    n = 4.264,
    l = 0.5,
    ks = 5.040,
    θr = 0.093,
    θs = 0.301,
    Ss = 1e-6,
    ψe = -1e-3,
)
sandspline = HDD.SplineConstitutive(sand)

millersand = HDD.RichardsCase(
    soil = sandspline,
    Δz = 0.0125,
    Δztotal = 10.0,
    tend = 0.18,
    ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
    topboundary = HDD.HeadBoundary(0.1, sandspline),
    bottomboundary = HDD.HeadBoundary(0.0, sandspline),
    forcing = nothing,
)
saveat = collect(0.0:0.01:millersand.tspan[2])

implicit_solver = HDD.NewtonSolver(
    HDD.LinearSolverThomas(millersand.parameters.n),
    relax = HDD.ScalarRelaxation(0.5),
)
implicit_result = HDD.benchmark!(
    HDD.implicit_model(millersand, implicit_solver, HDD.AdaptiveTimeStepper(0.1), saveat),
    millersand,
)

euler_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(
            alg = QBDF(nlsolve = NLNewton(relax = 0.5)),
            controller = HDD.CustomController(),
        ),
        saveat,
    ),
    millersand,
)


config = HDD.SolverConfig(
    alg = ImplicitEuler(nlsolve = NLNewton(relax = 0.5)),
    controller = HDD.CustomController(dtmin = 1e-6),
    maxiters = 100_000,
)

model = HDD.diffeq_model(millersand, config, saveat)

HDD.run!(model)




cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    millersand,
)

qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    millersand,
)

plot(implicit_result.model.saved[:, end])
plot!(qndf_result.model.saved[:, end])
plot!(cvode_result.model.saved[:, end])






qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    millersand,
)


qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    millersand,
)

cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    millersand,
)

using DASKR

daskr_solver = daskr(linear_solver = :Banded, jac_upper = 1, jac_lower = 1)
cvode_result = HDD.benchmark!(
    HDD.diffeq_model(millersand, HDD.SolverConfig(alg = daskr_solver), saveat),
    millersand,
)




sand = MualemVanGenuchten(
    a = 5.470,
    n = 4.264,
    m = 1.0 - 1 / 4.264,
    l = 0.5,
    ks = 5.040,
    θr = 0.093,
    θs = 0.301,
    Ss = 1e-6,
)
ψ = vcat(-exp10.(range(log10(10), log10(1e-3), length = 1000)), [1e-3, 2e-3])
sandspline = SplineConstitutive(
    PCHIPInterpolation(
        moisture_content.(ψ, Ref(sand)),
        ψ,
        extrapolation = DataInterpolations.ExtrapolationType.Constant,
    ),
    PCHIPInterpolation(
        conductivity.(ψ, Ref(sand)),
        ψ,
        extrapolation = DataInterpolations.ExtrapolationType.Constant,
    ),
    sand.θs,
    sand.Ss,
)
millersand = HDD.RichardsCase(
    soil = sandspline,
    Δz = 0.0125,
    Δztotal = 10.0,
    tend = 0.18,
    ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
    topboundary = HDD.HeadBoundary(0.1, sandspline),
    bottomboundary = HDD.HeadBoundary(0.0, sandspline),
    forcing = nothing,
)

cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    millersand,
)
