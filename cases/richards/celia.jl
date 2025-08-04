import HydroDaemonDemo as HDD
using DifferentialEquations

# Note: units are centimeters and seconds!
soil = HDD.Haverkamp(
    a = 1.611e6,
    B = 3.96,
    y = 4.74,
    A = 1.175e6,
    ks = 0.00944,
    θs = 0.287,
    θr = 0.075,
    Ss = 1e-6,
)
celia = HDD.RichardsCase(
    soil = soil,
    Δz = 1.0,
    Δztotal = 40.0,
    tend = 360.0,
    ψ0 = HDD.InitialConstant(-61.5),
    bottomboundary = HDD.HeadBoundary(-61.5, soil),
    topboundary = HDD.HeadBoundary(-20.5, soil),
    forcing = nothing,
)

function fixedΔt(celia, Δt, solver)
    saveat = collect(0.0:Δt:celia.tspan[2])
    return HDD.implicit_model(celia, solver, HDD.FixedTimeStepper(Δt), saveat)
end

implicit_solver = HDD.NewtonSolver(
    HDD.LinearSolverThomas(celia.parameters.n),
    relax = HDD.ScalarRelaxation(0.0),
)
implicit_result_01 = HDD.benchmark!(fixedΔt(celia, 0.1, implicit_solver), celia)
implicit_result_1 = HDD.benchmark!(fixedΔt(celia, 1.0, implicit_solver), celia)
implicit_result_10 = HDD.benchmark!(fixedΔt(celia, 10.0, implicit_solver), celia)
implicit_result_30 = HDD.benchmark!(fixedΔt(celia, 30.0, implicit_solver), celia)
implicit_result_120 = HDD.benchmark!(fixedΔt(celia, 120.0, implicit_solver), celia)

saveat = collect(0.0:1.0:celia.tspan[2])
euler_result = HDD.benchmark!(
    HDD.diffeq_model(
        celia,
        HDD.SolverConfig(alg = ImplicitEuler(), controller = HDD.CustomController()),
        saveat,
    ),
    celia,
)



qndf_result =
    HDD.benchmark!(HDD.diffeq_model(celia, HDD.SolverConfig(alg = QNDF()), saveat), celia)
qndf_dae_result = HDD.benchmark!(
    HDD.diffeq_model_dae(celia, HDD.SolverConfig(alg = QNDF()), saveat),
    celia,
)


diffeq_model = HDD.diffeq_model(celia, HDD.SolverConfig(alg = ImplicitEuler()), saveat)
HDD.run!(diffeq_model)

using BenchmarkTools
@btime HDD.reset_and_run!(diffeq_model, celia.ψ0)
