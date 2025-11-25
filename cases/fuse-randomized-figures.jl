using CSV
using DataFrames
using Plots
import HydroDaemonDemo as HDD


const COLORS = HDD.okabe_ito_colors()

function histogram(edges, data, allow_zero)
    nbins = length(edges) - 1
    counts = zeros(Int, nbins)
    for v in data
        idx = searchsortedfirst(edges, v) - 1
        if 1 <= idx <= nbins
            counts[idx] += 1
        end
    end
    relative_frequency = counts ./ length(data)
    if !allow_zero
        # We will plot with a log y-axis, so replace the zeros by a small number.
        relative_frequency[relative_frequency .== 0] .= 1e-10
    end
    return relative_frequency
end

function plot_fidelity!(p, fidelity_df)
    edges = collect(-1.5:0.05:1.5)
    centers = 0.5 * (edges[2:end] + edges[1:(end-1)])
    allow_zero = false

    plot!(
        p,
        centers,
        histogram(edges, fidelity_df[!, "Explicit"], allow_zero),
        ylim = (1e-4, 1.0),
        xlim = (-1.0, 1.0),
        yscale = :log10,
        lw = 2,
        label = "Explicit Euler",
        xlabel = "Fidelity (mm/d)",
        ylabel = "Frequency",
        color = COLORS[:dark_orange],
    )
    plot!(
        p,
        centers,
        histogram(edges, fidelity_df[!, "Implicit Newton"], allow_zero),
        lw = 2,
        label = "Implicit Euler",
        color = COLORS[:light_blue],
    )
    plot!(
        p,
        centers,
        histogram(edges, fidelity_df[!, "DiffEq-Tsit5"], allow_zero),
        lw = 2,
        label = "DiffEq-Tsit5",
        colors = COLORS[:green],
    )
    plot!(
        p,
        centers,
        histogram(edges, fidelity_df[!, "DiffEq-Rodas5P"], allow_zero),
        lw = 2,
        label = "DiffEq-Rodas5P",
        colors = COLORS[:pink],
    )
    return
end

function plot_timing!(p, timing_df)
    edges = collect(0.0:0.1:10.0)
    centers = 0.5 * (edges[2:end] + edges[1:(end-1)])
    allow_zero = true

    plot!(
        p,
        centers,
        histogram(edges, timing_df[!, "Explicit"], allow_zero),
        lw = 2,
        label = "Explicit Euler",
        xlabel = "Time (ms)",
        ylabel = "Frequency",
        color = COLORS[:dark_orange],
    )
    plot!(
        p,
        centers,
        histogram(edges, timing_df[!, "Implicit Newton"], allow_zero),
        lw = 2,
        label = "Implicit Euler",
        color = COLORS[:light_blue],
    )
    plot!(
        p,
        centers,
        histogram(edges, timing_df[!, "DiffEq-Tsit5"], allow_zero),
        lw = 2,
        label = "DiffEq-Tsit5",
        colors = COLORS[:green],
    )
    plot!(
        p,
        centers,
        histogram(edges, timing_df[!, "DiffEq-Rodas5P"], allow_zero),
        lw = 2,
        label = "DiffEq-Rodas5P",
        colors = COLORS[:pink],
    )
    return
end


fidelity_df = CSV.read("cases/output/fidelity_data.csv", DataFrame)
timing_df = CSV.read("cases/output/timing_data.csv", DataFrame) .* 1000.0

ptotal = plot(layout = (1, 2), size = (1200, 500), margin = 5Plots.mm)
plot_fidelity!(ptotal[1, 1], fidelity_df)
plot_timing!(ptotal[1, 2], timing_df)
display(current())

savefig("cases/output/fuse070-randomized.pdf")
