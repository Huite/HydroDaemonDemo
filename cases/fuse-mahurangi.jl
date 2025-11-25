import HydroDaemonDemo as HDD
using Dates
using Plots
using DifferentialEquations


const COLORS = HDD.okabe_ito_colors()

function multi_pane_plot(
    forcingdf,
    explicit_fuse,
    implicit_fuse,
    implicit_fuse_fine,
    diffeq_fuse,
    window,
)

    function flows(model, index)
        # Savedflows contain the time-integrated flows.
        return diff(model.savedflows[index, :])
    end

    dateformatter(x) = Dates.format(x, "dd-u")  # x is already a Date object
    mask = (window.start .<= forcingdf.Date .<= window.finish)
    tick_positions = forcingdf.Date[mask][1:20:end]
    tick_labels = dateformatter.(tick_positions)


    ptotal = plot(layout = grid(3, 1), size = (1000, 750), margin = 5Plots.mm)
    p1 = ptotal[1, 1]
    p2 = ptotal[2, 1]
    p3 = ptotal[3, 1]

    plot!(
        p1,
        forcingdf.Date[mask],
        forcingdf.Precipitation[mask];
        seriestype = :bar,
        color = :navy,
        alpha = 0.7,
        ylabel = "Precipitation\n(mm/day)",
        legend = false,
        xticks = (tick_positions, tick_labels),
    )

    output_t = forcingdf.Date
    start = findfirst(output_t .> window.start)
    finish = findlast(output_t .<= window.finish)
    t = output_t[start:finish]

    plot!(
        p2,
        t,
        implicit_fuse_fine.saved[1, start:finish];
        ylabel = "Unsat zone\nstorage (mm)",
        xticks = (tick_positions, tick_labels),
        legend = false,
        label = "Implicit Euler, Δt=1e-3",
        color = COLORS[:black],
        lw = 4,
    )
    plot!(
        p2,
        t,
        explicit_fuse.saved[1, start:finish];
        label = "Explicit Euler, Δt=1.0",
        color = COLORS[:dark_orange],
        lw = 2,
    )
    plot!(
        p2,
        t,
        implicit_fuse.saved[1, start:finish];
        label = "Implicit Euler, Δt=1.0",
        color = COLORS[:green],
        lw = 2,
    )
    plot!(
        p2,
        t,
        diffeq_fuse.saved[1, start:finish],
        label = "DifferentialEquations.jl",
        color = COLORS[:yellow],
        ls = :dash,
        lw = 2,
    )

    plot!(
        p3,
        t,
        flows(implicit_fuse_fine, 1)[start:finish];
        xlabel = "Date",
        ylabel = "Vertical drainage\n(mm/d)",
        xticks = (tick_positions, tick_labels),
        label = "Implicit Euler, Δt=1e-3",
        color = COLORS[:black],
        lw = 4,
        legend_columns = 4,
        legend = (0.1, -0.5),
        bottom_margin = 15Plots.mm,
    )
    plot!(
        p3,
        t,
        flows(explicit_fuse, 1)[start:finish];
        label = "Explicit Euler, Δt=1.0",
        color = COLORS[:dark_orange],
        lw = 2,
    )
    plot!(
        p3,
        t,
        flows(implicit_fuse, 1)[start:finish];
        label = "Implicit Euler, Δt=1.0",
        color = COLORS[:green],
        lw = 2,
    )
    plot!(
        p3,
        t,
        flows(diffeq_fuse, 1)[start:finish],
        label = "DiffEq-Tsit5",
        color = COLORS[:yellow],
        lw = 2,
        ls = :dash,
    )
    return
end


forcingdf = HDD.create_mahurangi_forcingdf(
    "data/mahurangi/*daily rainfall.csv",
    #    "data/1340__Evaporation__daily/1340__Evaporation__Priestly-Taylor-PET__daily.csv",
    "data/1340__Evaporation__daily/1340__Evaporation__Penman-Open-Water-Evaporation__daily.csv",
)

#figpath = "cases/output/mahurangi-1998-priestly-taylor.svg"
figpath = "cases/output/mahurangi-1998-penman-openwater.pdf"

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
    μτ = 0.6,       # d
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


implicit_fuse_fine = HDD.ImplicitHydrologicalModel(
    fuse,
    initial,
    solver,
    tspan,
    nothing,
    HDD.FixedTimeStepper(0.001),
)
HDD.run!(implicit_fuse_fine)

solverconfig = HDD.SolverConfig(alg = Tsit5())
diffeq_fuse = HDD.DiffEqHydrologicalModel(fuse, initial, tspan, nothing, solverconfig)
HDD.run!(diffeq_fuse)

window = (start = Date(1998, 6, 1), finish = Date(1998, 8, 31))
multi_pane_plot(
    forcingdf,
    explicit_fuse,
    implicit_fuse,
    implicit_fuse_fine,
    diffeq_fuse,
    window,
)
display(current())
savefig(figpath)
