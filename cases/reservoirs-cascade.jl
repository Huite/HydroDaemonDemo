using Plots
using DifferentialEquations
import HydroDaemonDemo as HDD
using Dates
using Statistics

forcingdf = HDD.create_mahurangi_forcingdf(
    "data/mahurangi/*daily rainfall.csv",
    "data/1340__Evaporation__daily/1340__Evaporation__Penman-Open-Water-Evaporation__daily.csv",
)
forcing = HDD.MeteorologicalForcing(
    0.0:1.0:(length(forcingdf.Date)-1),
    forcingdf.Precipitation,
    forcingdf.Evaporation .* 0.0,
)

n = 10
area = 1.0
a = 1.03
b = 1.0
cascade = HDD.BucketCascade(fill(area, n), fill(a, n), fill(b, n), forcing)
truncated_cascade = HDD.BucketCascade(fill(area, n), fill(a, n), fill(b, n), forcing, true)
initial = zeros(n)
tend = 120.0
tspan = (0.0, tend)

solverconfig = HDD.SolverConfig(alg = Tsit5())
diffeq_reservoirs =
    HDD.DiffEqHydrologicalModel(cascade, initial, tspan, nothing, solverconfig)
HDD.run!(diffeq_reservoirs)

explicit_reservoirs = HDD.ExplicitHydrologicalModel(
    cascade,
    initial,
    tspan,
    nothing,
    HDD.FixedTimeStepper(1.0),
)
HDD.run!(explicit_reservoirs)

explicit_truncated_reservoirs = HDD.ExplicitHydrologicalModel(
    truncated_cascade,
    initial,
    tspan,
    nothing,
    HDD.FixedTimeStepper(1.0),
)
HDD.run!(explicit_truncated_reservoirs)

solver = HDD.NewtonSolver(HDD.LinearSolverThomas(n), relax = HDD.SimpleLineSearch())
implicit_reservoirs = HDD.ImplicitHydrologicalModel(
    cascade,
    initial,
    solver,
    tspan,
    nothing,
    HDD.AdaptiveTimeStepper(Δt0 = 1.0),
)
HDD.run!(implicit_reservoirs)

q_analytical = HDD.analytical_solution(cascade, forcing, 1.0)[1:Int(tend)]
q_explicit = diff(explicit_reservoirs.savedflows[2, :])
q_trunc_explicit = diff(explicit_truncated_reservoirs.savedflows[2, :])
q_implicit = diff(implicit_reservoirs.savedflows[2, :])
q_diffeq = diff(diffeq_reservoirs.savedflows[2, :])


function plot_cascade(
    forcingdf,
    q_analytical,
    q_explicit,
    q_trunc_explicit,
    q_implicit,
    q_diffeq,
)
    figpath = "cases/output/reservoirs-flow.pdf"
    ptotal = plot(
        layout = grid(2, 1, heights = [0.45, 0.55]),
        size = (1000, 500),
        margin = 5Plots.mm,
    )
    p1 = ptotal[1, 1]
    p2 = ptotal[2, 1]

    colors = HDD.okabe_ito_colors()
    dateformatter(x) = Dates.format(x, "dd-u")
    x = forcingdf.Date[1:Int(tend)]
    tick_positions = forcingdf.Date[1:Int(tend)][1:15:end]
    tick_labels = dateformatter.(tick_positions)

    plot!(
        p1,
        forcingdf.Date[1:Int(tend)],
        forcingdf.Precipitation[1:Int(tend)];
        seriestype = :bar,
        color = :navy,
        alpha = 0.7,
        ylabel = "Precipitation\n(mm/day)",
        legend = false,
        xticks = (tick_positions, tick_labels),
    )

    plot!(
        p2,
        x,
        q_analytical,
        ylabel = "Outflow (mm/d)",
        xlabel = "Date",
        ylim = (-10.0, 20.0),
        xticks = (tick_positions, tick_labels),
        label = "Analytical Solution",
        color = colors[:black],
        legend = (0.2, -0.45),
        legend_columns = 3,
        lw = 5,
        bottom_margin = 20Plots.mm,
    )
    plot!(p2, x, q_explicit, label = "Explicit Euler", color = colors[:green], lw = 2)
    plot!(
        p2,
        x,
        q_trunc_explicit,
        label = "Truncated Explicit Euler",
        ls = :dash,
        lw = 2,
        color = colors[:blue],
    )
    plot!(p2, x, q_implicit, label = "Implicit Euler", color = colors[:dark_orange], lw = 3)
    plot!(
        p2,
        x,
        q_diffeq,
        label = "DifferentialEquations.jl",
        ls = :dash,
        color = colors[:yellow],
        lw = 2,
    )
    savefig(figpath)
    return
end

plot_cascade(forcingdf, q_analytical, q_explicit, q_trunc_explicit, q_implicit, q_diffeq)
display(current())

function rmse(numerical, exact)
    return sqrt(mean((numerical .- exact) .^ 2))
end

function nse(numerical, exact)
    μ = mean(exact)
    num = sum((numerical .- exact) .^ 2)
    den = sum((exact .- μ) .^ 2)
    return 1 - num / den
end


rmse_explicit = rmse(q_explicit, q_analytical)
rmse_trunc_explicit = rmse(q_trunc_explicit, q_analytical)
rmse_implicit = rmse(q_implicit, q_analytical)
rmse_diffeq = rmse(q_diffeq, q_analytical)

nse_explicit = nse(q_explicit, q_analytical)
nse_trunc_explicit = nse(q_trunc_explicit, q_analytical)
nse_implicit = nse(q_implicit, q_analytical)
nse_diffeq = nse(q_diffeq, q_analytical)
