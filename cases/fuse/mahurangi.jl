bounds = FuseParameterBounds(
    S1max = (lower = 25.0, upper = 500.0),
    S2max = (lower = 50.0, upper = 5000.0),
    ϕtens = (lower = 0.050, upper = 0.950),
    r1 = (lower = 0.050, upper = 0.950),
    ku = (lower = 0.010, upper = 1000.0),
    c = (lower = 1.0, upper = 20.0),
    ki = (lower = 0.010, upper = 1000.0),
    ks = (lower = 0.001, upper = 1000.0),
    n = (lower = 1.0, upper = 10.0),
    v = (lower = 0.001, upper = 0.250),
    Acmax = (lower = 0.050, upper = 0.950),
    b = (lower = 0.001, upper = 3.0),
    μτ = (lower = 0.010, upper = 5.0),
)

function realizations(; bounds, n)
end
