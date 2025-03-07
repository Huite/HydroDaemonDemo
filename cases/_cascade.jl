using HydroDaemonDemo
using CSV
using Dates
using DataFrames

const DAY = 86_400.0

df = CSV.read("cases/forcing.csv", DataFrame)
df.time = Dates.toms.(df.Date - df.Date[1]) / 1000.0

area = 100.0
n = 5
a = 1e-7
b = 1.5

cascade = HydroDaemonDemo.BucketCascade(
    fill(area, n),
    fill(a, n),
    fill(b, n),
    zeros(n),
    HydroDaemonDemo.PrecipitationForcing(df.time, df.P / 1000.0),
    HydroDaemonDemo.EvaporationForcing(df.time, df.ET / 1000.0),
)

tstart = 0.0 * DAY
tend = 300.0 * DAY
Δt = 0.01 * DAY
saveat = nothing


# Reset state
cascade.S .= 0.0
solver = HydroDaemonDemo.NewtonSolver(cascade, 100, 1e-6, 0.5)
implicit_output =
    HydroDaemonDemo.implicit_run!(cascade, solver, tstart, tend, Δt; saveat = saveat)


cascade.S .= 0.0
explicit_output = HydroDaemonDemo.explicit_run!(cascade, tstart, tend, Δt; saveat = saveat)

using Plots
plot(implicit_output[1, :])
plot!(explicit_output[1, :])
