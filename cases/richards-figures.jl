import HydroDaemonDemo as HDD
using DataFrames
using CSV
using Plots
using Dates

const COLORS = HDD.okabe_ito_colors()

celiadf = CSV.read("cases/output/celia-final-head.csv", DataFrame)
sanddf = CSV.read("cases/output/miller-sand-final-head.csv", DataFrame)
loamdf = CSV.read("cases/output/miller-loam-final-head.csv", DataFrame)
clayloamdf = CSV.read("cases/output/miller-clayloam-final-head.csv", DataFrame)

storagedf = CSV.read("cases/output/infiltration-storage.csv", DataFrame)
drainagedf = CSV.read("cases/output/infiltration-drainage.csv", DataFrame)


function plot_celia!(p, df)
    plot!(
        p,
        df[!, Symbol("Implicit Newton-0.1")],
        label = "Implicit Newton (Δt=0.1)",
        xlabel = "Elevation (cm)",
        ylabel = "Pressure head (cm)",
        color = COLORS[:black],
        lw = 3,
    )
    plot!(
        p,
        df[!, Symbol("Implicit Newton-10.0")],
        label = "Implicit Newton (Δt=10.0)",
        lw = 3,
        color = COLORS[:green],
        ls = :dash,
    )
    plot!(
        p,
        df[!, Symbol("Implicit Newton-30.0")],
        label = "Implicit Newton (Δt=30.0)",
        lw = 3,
        color = COLORS[:light_blue],
        ls = :dash,
    )
    plot!(
        p,
        df[!, Symbol("Implicit Newton-120.0")],
        label = "Implicit Newton (Δt=120.0)",
        lw = 3,
        color = COLORS[:orange],
        ls = :dash,
    )
    scatter!(p, df[!, Symbol("DiffEq-QNDF")], label = "DiffEq-QNDF", color = COLORS[:black])
    title!(p, "Celia benchmark")
    return
end

function plot_miller!(p, sanddf, loamdf, clayloamdf)
    function line_plot!(p, df, Δz, colname, label, color; ls = :solid)
        data = df[!, Symbol(colname)]
        n = length(data)
        z = collect((Δz*0.5):Δz:((n*Δz)-0.5*Δz))
        plot!(p, z, data, permute = (:y, :x), label = label, lw = 3, color = color, ls = ls)
        return
    end

    line_plot!(
        p,
        sanddf,
        0.0125,
        "Implicit Newton-0.0001",
        "Implicit Newton (Δtmax=0.0001)",
        COLORS[:black],
    )
    line_plot!(p, loamdf, 0.0125, "Implicit Newton-0.0001", nothing, COLORS[:black])
    line_plot!(p, clayloamdf, 0.00625, "Implicit Newton-0.0001", nothing, COLORS[:black])

    line_plot!(
        p,
        sanddf,
        0.0125,
        "DiffEq-CVODE_BDF",
        "Sand: DiffEq-CVODE_BDF",
        COLORS[:light_blue],
        ls = :dash,
    )
    line_plot!(
        p,
        loamdf,
        0.0125,
        "DiffEq-CVODE_BDF",
        "Loam: DiffEq-CVODE_BDF",
        COLORS[:orange],
        ls = :dash,
    )
    line_plot!(
        p,
        clayloamdf,
        0.00625,
        "DiffEq-CVODE_BDF",
        "Clay-loam: DiffEq-CVODE_BDF",
        COLORS[:green],
        ls = :dash,
    )


    ylabel!(p, "Elevation (m)")
    xlabel!(p, "Pressure head (m)")

    title!(p, "Miller benchmark")
    return
end

function plot_storage(p, df)
    start_date = Date("2004-01-01")
    end_date = Date("2004-12-31")
    date_range = collect(start_date:Day(1):end_date)
    start_index = findfirst(x -> x == start_date, df.date)
    end_index = findfirst(x -> x == end_date, df.date)

    plot!(
        p,
        date_range,
        df[start_index:end_index, Symbol("Implicit Newton")] .* 1000.0,
        label = "Implicit Newton (Δt=1.0)",
        xlabel = "Date",
        ylabel = "Storage (mm)",
        color = COLORS[:black],
        lw = 2,
        legend = :topleft,
    )
    plot!(
        p,
        date_range,
        df[start_index:end_index, Symbol("DiffEq-CVODE_BDF")] .* 1000.0,
        label = "DiffEq-CVODE_BDF",
        color = COLORS[:light_blue],
        lw = 2,
        ls = :dash,
    )
    title!(p, "Infiltration benchmark")
    return
end


function plot_drainage(p, df)
    start_date = Date("2004-01-01")
    end_date = Date("2004-12-31")
    date_range = collect(start_date:Day(1):end_date)
    start_index = findfirst(x -> x == start_date, df.date)
    end_index = findfirst(x -> x == end_date, df.date)

    plot!(
        p,
        date_range,
        df[start_index:end_index, Symbol("Implicit Newton")] .* 1000.0,
        label = "Implicit Newton (Δt=1.0)",
        xlabel = "Date",
        ylabel = "Drainage (mm/d)",
        color = COLORS[:black],
        lw = 2,
        legend = :topleft,
    )
    plot!(
        p,
        date_range,
        df[start_index:end_index, Symbol("DiffEq-QNDF")] .* 1000.0,
        label = "DiffEq-CVODE_BDF",
        color = COLORS[:light_blue],
        lw = 2,
        ls = :dash,
    )
    title!(p, "Infiltration benchmark")
    return

end

ptotal = plot(layout = (2, 2), size = (1200, 1000), margin = 5Plots.mm)
plot_celia!(ptotal[1, 1], celiadf)
plot_miller!(ptotal[1, 2], sanddf, loamdf, clayloamdf)
plot_storage(ptotal[2, 1], storagedf)
plot_drainage(ptotal[2, 2], drainagedf)

display(current())
savefig("cases/output/richards-results.pdf")
