import HydroDaemonDemo as HDD
using DifferentialEquations
using DataFrames
using Plots
using CSV

function create_celia()
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
        dt = 120.0,
        ψ0 = HDD.InitialConstant(-61.5),
        bottomboundary = HDD.HeadBoundary(-61.5, soil),
        topboundary = HDD.HeadBoundary(-20.5, soil),
        forcing = nothing,
    )
    return celia
end

function run(case, solver_presets)
    rows = []
    results = []
    for preset in solver_presets
        if preset isa HDD.ImplicitSolverPreset
            name = "$(HDD.name(preset))-$(preset.timestepper.Δt0)"
        else
            name = HDD.name(preset)
        end
        println("Benchmarking Celia, $(HDD.name(preset))")
        result = HDD.benchmark!(case, preset)
        push!(results, result)
        push!(
            rows,
            (
                solver = name,
                time = minimum(result.trial).time / 1e9,
                mass_bias = result.mass_bias,
                mass_rmse = result.mass_rsme,
            ),
        )
    end
    df = DataFrame(rows)
    return df, results
end


celia = create_celia()
solver_presets = (
    #HDD.ImplicitSolverPreset(timestepper=HDD.FixedTimeStepper(0.1)),
    #HDD.ImplicitSolverPreset(timestepper=HDD.FixedTimeStepper(1.0)),
    #HDD.ImplicitSolverPreset(timestepper=HDD.FixedTimeStepper(10.0)),
    #HDD.ImplicitSolverPreset(timestepper=HDD.FixedTimeStepper(30.0)),
    #HDD.ImplicitSolverPreset(timestepper=HDD.FixedTimeStepper(120.0)),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = ImplicitEuler())),
    #HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = ImplicitEuler())),
    HDD.DiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
    #HDD.DAEDiffEqSolverPreset(HDD.SolverConfig(alg = QNDF())),
)

df, results = run(celia, solver_presets)
CSV.write("cases/richards/celia.csv", df)


function plothead(case, solver_presets, results)
    plot()
    n = case.parameters.n
    for (preset, result) in zip(solver_presets, results)

        if preset isa HDD.ImplicitSolverPreset
            name = "$(HDD.name(preset))-$(preset.timestepper.Δt0)"
        else
            name = HDD.name(preset)
        end

        ψ = result.model.saved[1:n, end]
        plot!(ψ, label = name, xlabel = "Depth (cm)", ylabel = "Pressure head (cm)")
    end
    display(current())
    return
end

plothead(celia, solver_presets, results)
savefig("cases/richards/celia.png")
savefig("cases/richards/celia.svg")
