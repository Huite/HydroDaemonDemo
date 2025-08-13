import HzsydroDaemonDemo as HDD
using DifferentialEquations
using Sundials
using DataFrames
using CSV
using Plots


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
    for (soil, case) in pairs(cases)
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
#    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 500_000)),
#    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = ImplicitEuler(), maxiters = 500_000)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 200_000)),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 200_000)),
#    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = FBDF(), maxiters = 200_000)),
#    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = FBDF(), maxiters = 500_000)),  # 2e5 not enough
#    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QBDF(), maxiters = 200_000)),
#    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QBDF(), maxiters = 200_000)),
#    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1))),
)
cases = (
    sand = create_millersand(),
    loam = create_millerloam(),
    clayloam = create_millerclayloam(),
)

df, results = run(cases, solver_presets)
CSV.write("cases/richards/miller.csv", df)


function plotresult(cases, solver_presets, results)
    plot()
    for ((preset, (soil, case)), result) in zip(Iterators.product(solver_presets, pairs(cases)), results)
        n = case.parameters.n
        Δz = case.parameters.Δz
        z = collect(Δz:Δz:n*Δz)
        ψ = result.model.saved[1:n, end]
        plot!(
            z,
            ψ,
            permute=(:y, :x),
            label="$(string(soil)), $(HDD.name(preset))",
            ylabel="Pressure head (m)",
            xlabel="Elevation (m)",
        )
    end
    display(current())
    return
end

plotresult(cases, solver_presets, results)
savefig("cases/richards/miller.svg")