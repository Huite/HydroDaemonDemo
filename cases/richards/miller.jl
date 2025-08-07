import HydroDaemonDemo as HDD
using DifferentialEquations
using Sundials
using DataFrames
using CSV


function create_millersand()
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
        dt = 0.01,
        ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
        topboundary = HDD.HeadBoundary(0.1, sandspline),
        bottomboundary = HDD.HeadBoundary(0.0, sandspline),
        forcing = nothing,
    )
    return millersand
end

function create_millerloam()
    loam = HDD.ModifiedMualemVanGenuchten(
        a = 3.600,
        n = 1.560,
        l = 0.5,
        ks = 0.250,
        θr = 0.078,
        θs = 0.430,
        Ss = 1e-6,
        ψe = -1e-3,
    )
    loamspline = HDD.SplineConstitutive(loam)
    millerloam = HDD.RichardsCase(
        soil = loamspline,
        Δz = 0.0125,
        Δztotal = 5.0,
        tend = 2.25,
        dt = 0.01,
        ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
        topboundary = HDD.HeadBoundary(0.1, loamspline),
        bottomboundary = HDD.HeadBoundary(0.0, loamspline),
        forcing = nothing,
    )
    return millerloam
end

function create_millerclayloam()
    clayloam = HDD.ModifiedMualemVanGenuchten(
        a = 1.900,
        n = 1.310,
        l = 0.5,
        ks = 0.062,
        θr = 0.095,
        θs = 0.410,
        ψe = -1e-3,
        Ss = 1e-6,
    )
    clayloamspline = HDD.SplineConstitutive(clayloam)

    millerclayloam = HDD.RichardsCase(
        soil = clayloamspline,
        Δz = 0.00625,
        Δztotal = 2.0,
        tend = 1.0,
        dt = 0.01,
        ψ0 = HDD.InitialHydrostatic(watertable = 0.0),
        topboundary = HDD.HeadBoundary(0.1, clayloamspline),
        bottomboundary = HDD.HeadBoundary(0.0, clayloamspline),
        forcing = nothing,
    )
    return millerclayloam
end

function run(cases, solver_presets)
    results = []
    rows = []
    for (soil, case) in cases
        for preset in solver_presets
            println("Benchmarking $(soil), $(HDD.name(preset))")
            result = HDD.benchmark!(case, preset)
            push!(results, result)
            push!(
                rows,
                (
                    soil = string(soil),
                    solver = HDD.name(preset),
                    time = minimum(result.trial).time / 1e9,
                    mass_bias = result.mass_bias,
                    mass_rmse = result.mass_rsme,
                ),
            )
        end
    end

    df = DataFrame(rows)
    return df, results
end

solver_presets = (
    HDD.ImplicitSolverPreset(
        relax = 0.0,
        abstol = 1e-6,
        reltol = 1e-6,
        timestepper = HDD.AdaptiveTimeStepper(0.01),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 200_000)),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 200_000)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = FBDF(), maxiters = 200_000)),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = FBDF(), maxiters = 300_000)),  # 2e5 not enough
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QBDF(), maxiters = 200_000)),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QBDF(), maxiters = 200_000)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1))),
)
cases = (
    sand = create_millersand(),
    loam = create_millerloam(),
    clayloam = create_millerclayloam(),
)

df, results = run(cases, solver_presets)

df = DataFrame(rows)

result = HDD.benchmark!(cases[:sand], solver_presets[1])


millersand = cases[:sand]

implicit_result = HDD.benchmark!(
    HDD.implicit_model(millersand, implicit_solver, HDD.AdaptiveTimeStepper(0.1), saveat),
    millersand,
) # relax 0.5 :370 ms, relax 0.0: 532 ms
# relax 0.5, reltol 1e-3: 254 ms


solution = "Implicit Newton"
soil = "Sand"
time = minimum(implicit_result.trial.time) / 1e9
mass_bias = implicit_result.mass_bias
mass_rmse = implicit_result.mass_rsme


qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = QNDF(), maxiters = 200_000, reltol = 1e-6),
        saveat,
    ),
    millersand,
)  # heads-based: 16.7 s; dae: 12.3 s

qndf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millersand,
        HDD.SolverConfig(alg = QNDF(), maxiters = 200_000, reltol = 1e-6),
        saveat,
    ),
    millersand,
)  # heads-based: 16.7 s; dae: 12.3 s
# reltol 1e-3: 6 s

qbdf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millersand,
        HDD.SolverConfig(alg = QBDF(), maxiters = 200_000),
        saveat,
    ),
    millersand,
)  # 24 s, dae: 13.5 s


fbdf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = FBDF(), maxiters = 200_000),
        saveat,
    ),
    millersand,
)  # dae: 35.2 s


cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        millersand,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    millersand,
)  # really quite slow

euler_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millersand,
        HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 500_000, reltol = 1e-3),
        saveat,
    ),
    millersand,
)  # reltol 1e-3: 11.1 s 


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



implicit_solver = HDD.NewtonSolver(
    HDD.LinearSolverThomas(millerclayloam.parameters.n),
    relax = HDD.ScalarRelaxation(0.0),
)
implicit_result = HDD.benchmark!(
    HDD.implicit_model(
        millerclayloam,
        implicit_solver,
        HDD.AdaptiveTimeStepper(0.1),
        saveat,
    ),
    millerclayloam,
)  # relax 0.5: 124 ms, relax 0.0: 50 ms

cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        millerclayloam,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    millerclayloam,
)  # 5.9 s
euler_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millerclayloam,
        HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 100_000),
        saveat,
    ),
    millerclayloam,
)  # 4.1 s, dae: 8.4 s


euler_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millerclayloam,
        HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 100_000),
        saveat,
    ),
    millerclayloam,
)  # 4.1 s, dae: 8.4 s



euler_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millerclayloam,
        HDD.SolverConfig(
            alg = ImplicitEuler(),
            maxiters = 100_000,
            controller = HDD.CustomController(dtmin = 1e-9),
        ),
        saveat,
    ),
    millerclayloam,
)  # 4.1 s, dae: 8.4 s

qndf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millerclayloam,
        HDD.SolverConfig(alg = QNDF(), maxiters = 100_000),
        saveat,
    ),
    millerclayloam,
)  # 4.1 s, dae: 11.2 s

qbdf_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        millerclayloam,
        HDD.SolverConfig(alg = QBDF(), maxiters = 100_000),
        saveat,
    ),
    millerclayloam,
)  # 4.1 s, dae: 9.5 s
fbdf_result = HDD.benchmark!(
    HDD.diffeq_model(
        millerclayloam,
        HDD.SolverConfig(alg = FBDF(), maxiters = 100_000),
        saveat,
    ),
    millerclayloam,
)
