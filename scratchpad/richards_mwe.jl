using DifferentialEquations
using Plots
using LinearAlgebra


struct Haverkamp
    # Haverkamp
    α::Float64
    β::Float64
    γ::Float64
    A::Float64
    ks::Float64
    θs::Float64
    θr::Float64
    # Specific storage coefficient
    Ss::Float64
    # Boundary conditions
    ψtop::Float64
    ψbot::Float64
    # Geometry
    Δz::Float64
    # Number of cells
    n::Int
end

function Haverkamp(; α, β, γ, A, ks, θs, θr, SS, ψtop, ψbot, Δz, n)
    return Haverkamp(α, β, γ, A, ks, θs, θr, SS, ψtop, ψbot, Δz, n)
end

"""k(ψ)"""
function conductivity(ψ, h::Haverkamp)
    return h.ks * h.A / (h.A + abs(min(ψ, 0.0))^h.γ)
end

"""θ(ψ)"""
function moisture_content(ψ, h::Haverkamp)
    return h.α * (h.θs - h.θr) / (h.α + abs(ψ)^h.β) + h.θr
end

"""dθ/dψ"""
function specific_moisture_capacity(ψ, h::Haverkamp)
    return h.α * h.β * (h.θs - h.θr) * abs(ψ)^(h.β - 1) / (h.α + abs(ψ)^h.β)^2
end

function bottomflux(ψ, p, k)
    boundary_k = conductivity(p.ψbot, p)
    kmean = 0.5 * (k[1] + boundary_k)
    Δψ = p.ψbot - ψ[1]
    Δz = 0.5 * p.Δz
    return kmean * (Δψ / Δz - 1)
end

function topflux(ψ, p, k)
    boundary_k = conductivity(p.ψtop, p)
    kmean = 0.5 * (k[end] + boundary_k)
    Δψ = p.ψtop - ψ[end]
    Δz = 0.5 * p.Δz
    return kmean * (Δψ / Δz + 1)
end

function storage_coefficient(ψ, p)
    C = specific_moisture_capacity(ψ, p)
    return p.Δz * C
end

function massmatrix_richards!(du, u, p, t)
    ψu = @view u[1:p.n]
    θu = @view u[p.n+1:end]
    k = [conductivity(ψ, p) for ψ in ψu]
    Δψ = @views ψu[2:end] .- ψu[1:end-1]
    k_inter = @views 0.5 .* (k[2:end] .+ k[1:end-1])
    kΔψΔz⁻¹ = Δψ .* k_inter ./ p.Δz

    # Internodal flows
    ∇q = @view du[1:p.n]
    ∇q .= 0.0
    @views @. ∇q[2:end] -= (kΔψΔz⁻¹ + k_inter)
    @views @. ∇q[1:end-1] += (kΔψΔz⁻¹ + k_inter)

    # Boundary conditions
    ∇q[1] += bottomflux(ψu, p, k)
    ∇q[end] += topflux(ψu, p, k)

    # Divide by C + SS
    Ss = [storage_coefficient(ψ, p) for ψ in ψu]
    @views ∇q ./= SS

    # Formulate algebraic constraint
    dθ = @view du[p.n+1:end]
    θ_ψ = [moisture_content(ψ, p) for ψ in ψu]
    @. dθ = θu - θ_ψ
    return
end

n = 40
p = Haverkamp(
    α = 1.611e6,
    β = 3.96,
    γ = 4.74,
    A = 1.175e6,
    ks = 0.00944,
    θs = 0.287,
    θr = 0.075,
    Ss = 1e-6,
    ψtop = -20.5,
    ψbot = -61.5,
    Δz = 1.0,
    n = n,
)
tspan = (0.0, 360.0)

u0 = [fill(-61.5, n); [moisture_content(-61.5, p) for _ = 1:n]]
M = Diagonal([ones(n); zeros(n)])
ode_function = ODEFunction(massmatrix_richards!, mass_matrix = M)
ode_problem = ODEProblem(ode_function, u0, tspan, p)
out = solve(ode_problem, ImplicitEuler(autodiff = true))
plot(out[end][1:n])
#plot(out[end][n+1:end])  # θ

u0 = [fill(-61.5, n); [moisture_content(-61.5, p) for _ = 1:n]]
M = Diagonal([ones(n); zeros(n)])
ode_function = ODEFunction(massmatrix_richards!, mass_matrix = M)
ode_problem = ODEProblem(ode_function, u0, tspan, p)
out = solve(ode_problem, QNDF(autodiff = true))
plot!(out[end][1:n])
#plot(out[end][n+1:end])


function init_and_run(alg)
    u0 = [fill(-61.5, n); [moisture_content(-61.5, p) for _ = 1:n]]
    M = Diagonal([ones(n); zeros(n)])
    ode_function = ODEFunction(massmatrix_richards!, mass_matrix = M)
    ode_problem = ODEProblem(ode_function, u0, tspan, p)
    out = solve(ode_problem, alg)
    return
end

using BenchmarkTools

@btime init_and_run(ImplicitEuler())
@btime init_and_run(QNDF())
@btime init_and_run(Rodas4())



# These probably need explicit annotation of differential_vars (vs algebraic)
# And explicit DAEProblem instead of ODEProblem?

using Sundials

@btime init_and_run(IDA())

using DASKR

@btime init_and_run(IDA())





function headbased_richards!(du, u, p, t)
    C = [specific_moisture_capacity(ψ, p) for ψ in u]
    k = [conductivity(ψ, p) for ψ in u]
    Δψ = @views u[2:end] .- u[1:end-1]
    k_inter = @views 0.5 .* (k[2:end] .+ k[1:end-1])
    kΔψΔz⁻¹ = Δψ .* k_inter ./ p.Δz

    # Internodal flows
    ∇q = du
    ∇q .= 0.0
    @views @. ∇q[2:end] -= (kΔψΔz⁻¹ + k_inter)
    @views @. ∇q[1:end-1] += (kΔψΔz⁻¹ + k_inter)

    # Boundary conditions
    ∇q[1] += bottomflux(u, p, k)
    ∇q[end] += topflux(u, p, k)

    @. du = ∇q / (p.Δz * (C + p.SS))
    return
end

ode_function = ODEFunction(headbased_richards!)
ode_problem = ODEProblem(ode_function, u0, tspan, p)
out1 = solve(ode_problem, QBDF1())
plot(out1[end])


function mass_matrix_value(ψ, p)
    C = specific_moisture_capacity(ψ, p)
    return p.Δz * (C + p.SS)
end

function update_mm!(L, u, p, t)
    println("updating mass matrix")
    for (i, ψ) in enumerate(u)
        L[i] = mass_matrix_value(ψ, p)
    end
    return
end
