using DASKR
import HydroDaemonDemo as HDD
using Plots

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

function Haverkamp(; α, β, γ, A, ks, θs, θr, Ss, ψtop, ψbot, Δz, n)
    return Haverkamp(α, β, γ, A, ks, θs, θr, Ss, ψtop, ψbot, Δz, n)
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

function dae_richards!(residual, du, u, p, t)
    @views dψ = du
    @views ψ = u
    @views F = residual

    k = [conductivity(ψi, p) for ψi in ψ]
    Δψ = @views ψ[2:end] .- ψ[1:(end-1)]
    k_inter = @views 0.5 .* (k[2:end] .+ k[1:(end-1)])
    kΔψΔz⁻¹ = Δψ .* k_inter ./ p.Δz

    ∇q = zero(F)
    @views @. ∇q[2:end] -= (kΔψΔz⁻¹ + k_inter)
    @views @. ∇q[1:(end-1)] += (kΔψΔz⁻¹ + k_inter)

    # Boundary conditions
    ∇q[1] += bottomflux(ψ, p, k)
    ∇q[end] += topflux(ψ, p, k)

    # Now construct the DAE residual correctly
    for i = 1:p.n
        C = specific_moisture_capacity(ψ[i], p)
        F[i] = -(∇q[i] - p.Δz * (C + p.Ss) * dψ[i])
    end
    return
end

function create_daskr_richards(p)
    function richards_daskr(t, y, yp, res)
        # Convert DASKR signature (t, y, yp, res) to your signature (res, yp, y, p, t)
        dae_richards!(res, yp, y, p, t)
        return nothing
    end
    return richards_daskr
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
u0 = fill(-61.5, n)
du0 = zeros(n)
f = create_daskr_richards(p)

y = copy(u0)
yp = copy(du0)
id = Int32.(ones(length(y)))  # All differential variables
tstart = 0.0
tstop = 360.0
Nsteps = 360
abstol = 1e-6
reltol = 1e-6

tstep = tstop / Nsteps
tout = [tstep]
idid = Int32[0]
info = zeros(Int32, 20)

info[11] = 0
info[16] = 0    # == 1 to ignore algebraic variables in the error calculation
info[17] = 0
info[18] = 2    # more initialization info
N = Int32[length(y)]
t = [tstart]
nrt = Int32[0]
rpar = [0.0]
rtol = [reltol]
atol = [abstol]
lrw = Int32[N[1]^3+9*N[1]+60+3*nrt[1]]
rwork = zeros(lrw[1])
liw = Int32[2*N[1]+40]
iwork = zeros(Int32, liw[1])
iwork[40 .+ (1:N[1])] = id
jroot = zeros(Int32, max(nrt[1], 1))
ipar = Int32[length(y), nrt[1], length(y)]
res = DASKR.res_c(f)
rt = Int32[0]
jac = Int32[0]
psol = Int32[0]

sol_t = Float64[]
sol_y = Vector{Float64}[]
sol_yp = Vector{Float64}[]

push!(sol_t, t[1])
push!(sol_y, copy(y))
push!(sol_yp, copy(yp))

# Time stepping loop
for step = 1:Nsteps
    DASKR.unsafe_solve(
        res,
        N,
        t,
        y,
        yp,
        tout,
        info,
        rtol,
        atol,
        idid,
        rwork,
        lrw,
        iwork,
        liw,
        rpar,
        ipar,
        jac,
        psol,
        rt,
        nrt,
        jroot,
    )

    if idid[1] < 0
        error("DASKR failed with idid = $(idid[1])")
    end

    # Store solution
    push!(sol_t, t[1])
    push!(sol_y, copy(y))
    push!(sol_yp, copy(yp))

    # Update tout for next step
    if step < Nsteps
        tout[1] = tstart + (step + 1) * tstep
    end
end



