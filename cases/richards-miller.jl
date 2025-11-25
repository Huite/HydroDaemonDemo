import HydroDaemonDemo as HDD
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
        soil = sand,#spline,
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
        soil = loam,#spline,
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
        soil = clayloam,#spline,
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
    HDD.ImplicitNewtonSolverPreset(
        relax = HDD.SimpleLineSearch(),
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 0.01),
    ),
    HDD.ImplicitNewtonSolverPreset(
        relax = HDD.SimpleLineSearch(),
        timestepper = HDD.AdaptiveTimeStepper(Δt0 = 0.0001, Δtmax = 0.0001),
    ),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 199_000)),
    HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF(), maxiters = 200_000)),
    HDD.DiffEqSolverPreset(
        HDD.SolverConfig(
            alg = CVODE_BDF(linear_solver = :Band, jac_upper = 1, jac_lower = 1),
        ),
    ),
)
cases = (
    sand = create_millersand(),
    loam = create_millerloam(),
    clayloam = create_millerclayloam(),
)

df, results = run(cases, solver_presets)
CSV.write("cases/output/miller.csv", df)

n = length(solver_presets)
sand_results = results[1:n]
loam_results = results[(n+1):(2*n)]
clayloam_results = results[(2*n+1):(n*3)]
soils = ["sand", "loam", "clayloam"]

for (soil, soil_results) in zip(soils, (sand_results, loam_results, clayloam_results))
    data = Dict{String,Vector{Float64}}()
    for (preset, result) in zip(solver_presets, soil_results)
        if (preset isa HDD.ImplicitNewtonSolverPreset) |
           (preset isa HDD.ImplicitPicardSolverPreset)
            name = "$(HDD.name(preset))-$(preset.timestepper.Δt0)"
        else
            name = HDD.name(preset)
        end

        model = result.model
        parameters = HDD.get_parameters(model)
        finalψ = model.saved[1:parameters.n, end]

        data[name] = finalψ
    end

    headdf = DataFrame(data)
    CSV.write("cases/output/miller-$(soil)-final-head.csv", headdf)
end
