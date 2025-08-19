import HydroDaemonDemo as HDD
using Dates
using Plots
using DifferentialEquations


function multi_pane_plot(forcingdf, explicit_fuse, implicit_fuse, diffeq_fuse, window)
    function flows(model, index)
        # Savedflows contain the time-integrated flows.
        return diff(model.savedflows[index, :])
    end

    dateformatter(x) = Dates.format(x, "dd-u")  # x is already a Date object
    mask = (window.start .<= forcingdf.Date .<= window.finish)
    tick_positions = forcingdf.Date[mask][1:20:end]
    tick_labels = dateformatter.(tick_positions)

    p1 = plot(
        forcingdf.Date[mask],
        forcingdf.Precipitation[mask];
        seriestype = :bar,
        color = :navy,
        alpha = 0.7,
        ylabel = "Precipitation\n(mm/day)",
        legend = false,
        size = (1000, 250),
        xticks = (tick_positions, tick_labels),
    )

    output_t = forcingdf.Date + Dates.Day(1)
    start = findfirst(output_t .> window.start)
    finish = findlast(output_t .<= window.finish)
    t = output_t[start:finish]

    p2 = plot(
        t,
        explicit_fuse.saved[1, start:finish];
        ylabel = "Unsat zone storage (mm)",
        xticks = (tick_positions, tick_labels),
        legend = false,
        label = "Explicit Euler",
    )
    plot!(p2, t, implicit_fuse.saved[1, start:finish]; label = "Implicit Euler")
    plot!(p2, t, diffeq_fuse.saved[1, start:finish], label = "DifferentialEquations.jl")

    p3 = plot(
        t,
        flows(explicit_fuse, 1)[start:finish];
        xlabel = "Date",
        ylabel = "Vertical drainage (mm/d)",
        xticks = (tick_positions, tick_labels),
        label = "Explicit Euler",
    )
    plot!(p3, t, flows(implicit_fuse, 1)[start:finish]; label = "Implicit Euler")
    plot!(p3, t, flows(diffeq_fuse, 1)[start:finish], label = "DifferentialEquations.jl")
    result = plot(p1, p2, p3, layout = (3, 1), size = (1000, 750), margin = 5Plots.mm)
    return result
end


forcingdf = HDD.create_mahurangi_forcingdf(
    "data/mahurangi/*daily rainfall.csv",
    #"data/1340__Evaporation__daily/1340__Evaporation__Priestly-Taylor-PET__daily.csv",
    #    "data/1340__Evaporation__daily/1340__Evaporation__Raised-Pan__daily.csv",
    #    "data/1340__Evaporation__daily/1340__Evaporation__Penman-PET__daily.csv",
    "data/1340__Evaporation__daily/1340__Evaporation__Penman-Open-Water-Evaporation__daily.csv",
)

forcing = HDD.MeteorologicalForcing(
    0.0:1.0:(length(forcingdf.Date)-1),
    forcingdf.Precipitation,
    forcingdf.Evaporation,
)
fuse = HDD.Fuse070Parameters(
    ϕtens = 0.5,    # 
    S1max = 100.0,  # mm
    b = 0.2,        # 
    ku = 500.0,     # mm/d
    c = 10,         # 
    v = 0.1,        # 1/d
    forcing = forcing,
)
initial = zeros(2)
tspan = (forcing.t[1], forcing.t[end] + 1.0)

explicit_fuse =
    HDD.ExplicitHydrologicalModel(fuse, initial, tspan, nothing, HDD.FixedTimeStepper(1.0))
HDD.run!(explicit_fuse)

solver = HDD.NewtonSolver(HDD.LinearSolverThomas(2), relax = HDD.SimpleLineSearch())
implicit_fuse = HDD.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HDD.FixedTimeStepper(1.0),
)
HDD.run!(implicit_fuse)

solver = HDD.NewtonSolver(HDD.LinearSolverThomas(2), relax = HDD.SimpleLineSearch())
implicit_fuse = HDD.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HDD.FixedTimeStepper(1.0),
)
HDD.run!(implicit_fuse)

solverconfig = HDD.SolverConfig(alg = Tsit5())
diffeq_fuse = HDD.DiffEqHydrologicalModel(fuse, initial, tspan, nothing, solverconfig)
HDD.run!(diffeq_fuse)

window = (start = Date(1998, 6, 1), finish = Date(1998, 8, 31))
multi_pane_plot(forcingdf, explicit_fuse, implicit_fuse, diffeq_fuse, window)
savefig("cases/output/mahurangi-1998-penman-openwater.svg")
#savefig("cases/output/mahurangi-1998-penman-PET.svg")
#savefig("cases/output/mahurangi-1998-raised-pan.svg")
#savefig("cases/output/mahurangi-1998-priestly-taylor.svg")
