using HydroDaemonDemo
using DataInterpolations
using Plots

constitutive = HydroDaemonDemo.MualemVanGenuchten(
    a = 1.900,
    n = 1.310,
    l = 0.5,
    ks = 0.062,
    θr = 0.095,
    θs = 0.410,
)

constitutive = HydroDaemonDemo.Haverkamp(
    a = 1.611e6,
    β = 3.96,
    y = 4.74,
    A = 1.175e6,
    ks = 0.00944,  # m/s
    θs = 0.287,
    θr = 0.075,
)


hermite, error = HydroDaemonDemo.HermiteSplineConstitutive(constitutive, ψmin=-1e3)
ψ_sample = collect(-11e-3:2e-3:4e-3)

plot(hermite.k, yscale=:log10)
plot(hermite.θ)

ψ_check = collect(-12e-3:1e-4:4e-3)
u_check = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in ψ_check]
plot(ψ_check, u_check)
scatter!(ψ_check, hermite.k.(ψ_check))


function check_finite_diff(ψ, eps, constitutive)
    return [(HydroDaemonDemo.conductivity(v+eps, constitutive) - HydroDaemonDemo.conductivity(v, constitutive)) / eps for v in ψ]
end

check_finite_diff(ψ_sample, 1e-6, constitutive)

constitutive = HydroDaemonDemo.MualemVanGenuchten(
    a = 1.900,
    n = 1.310,
    l = 0.5,
    ks = 0.062,
    θr = 0.095,
    θs = 0.410,
    ψe = 0.02,
)

ψ_min = -1e4
ψ_e = -0.02
knots = vcat(
    -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=100)),
    collect(ψ_e:1e-3:1e-2),
)

u = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in knots]
interp = PCHIPInterpolation(u, knots)
plot(interp)

ψ_check = collect(-12e-3:1e-4:3e-2) .- 0.02
u_check = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in ψ_check]
plot(ψ_check, u_check)
ψ_interp = collect(-12e-3:1e-3:3e-2) .- 0.02
u_interp = [interp(ψ) for ψ in ψ_interp]
scatter!(ψ_interp, u_interp)


test_knots = vcat(
    -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=300)),

)

error = [interp(p) - HydroDaemonDemo.conductivity(p, constitutive) for p in test_knots]
relerror = [(interp(p) - HydroDaemonDemo.conductivity(p, constitutive)) / HydroDaemonDemo.conductivity(p, constitutive) for p in test_knots]
extrema(relerror)


constitutive = HydroDaemonDemo.MualemVanGenuchten(
    a = 1.900,
    n = 1.310,
    l = 0.5,
    ks = 0.062,
    θr = 0.095,
    θs = 0.410,
    ψe = -0.02,
)

ψ_min = -1e4
ψ_e = -0.02
knots = vcat(
    -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=50)),
    0.0,
)

u = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in knots]
interp = PCHIPInterpolation(log.(u), knots)
plot(interp)

u_check = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in ψ_check]
plot(ψ_check, u_check)
ψ_interp = collect(-12e-3:1e-3:0e-2) .- 0.02
u_interp = exp.([interp(ψ) for ψ in ψ_interp])
scatter!(ψ_interp, u_interp)

test_knots = vcat(
    -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=300)),

)
relerror = Float64[]
for p in test_knots
    check = HydroDaemonDemo.conductivity(p, constitutive)
    error = exp(interp(p)) - check
    rel = error / check
    push!(relerror, rel)
end
extrema(relerror)


scatter(knots, log.(u))



function create_pchip_interpolator(ψ_min, ψ_e, n, constitutive)
    knots = vcat(
        -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=n)),
    )
    u = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in knots]
    interp = PCHIPInterpolation(u, knots)
    return interp
end

function create_interpolator(ψ_min, ψ_e, n, constitutive)
    knots = vcat(
        -exp10.(range(log10(abs(ψ_min)), log10(abs(ψ_e)), length=n)),
    )
    u = [HydroDaemonDemo.conductivity(ψ, constitutive) for ψ in knots]
    du = [HydroDaemonDemo.dconductivity(ψ, constitutive) for ψ in knots]
    interp = CubicHermiteSpline(du, u, knots)
    return interp
end

pchip = create_pchip_interpolator(-1e4, -0.02, 50, constitutive)
plot(pchip, yscale=:log10)

DataInterpolations.derivative(pchip, -0.03)
HydroDaemonDemo.dconductivity(-0.03, constitutive)

hspline = create_interpolator(-1e4, -0.02, 50, constitutive)
plot(hspline, yscale=:log10)
DataInterpolations.derivative(hspline, -0.03)
HydroDaemonDemo.dconductivity(-0.03, constitutive)


t = [-0.03, 0.0]
u = HydroDaemonDemo.conductivity.(t, Ref(constitutive))
du = HydroDaemonDemo.dconductivity.(t, Ref(constitutive))
local_interpolation = CubicHermiteSpline(du, u, t)

test = collect(-0.03:1e-3:0.0)
y = local_interpolation.(test)
plot(test, y)