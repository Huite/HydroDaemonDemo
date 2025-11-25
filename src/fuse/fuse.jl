abstract type FuseParameters <: Parameters end

const FUSE_RHO = 0.01

# [core]
struct Fuse070Parameters <: FuseParameters
    ϕtens::Float64
    S1max::Float64
    b::Float64
    ku::Float64
    c::Float64
    v::Float64
    μτ::Float64
    ω::Float64
    n::Int
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
    function Fuse070Parameters(; ϕtens, S1max, b, ku, c, v, μτ, forcing)
        ω = S1max * FUSE_RHO  # ρ from FUSE paper
        nstate = 2
        new(ϕtens, S1max, b, ku, c, v, μτ, ω, nstate, forcing, zeros(2))
    end
end

function Base.show(io::IO, p::Fuse070Parameters)
    println(io, "Fuse070Parameters:")
    println(io, "  ϕtens: $(p.ϕtens)")
    println(io, "  S1max: $(p.S1max)")
    println(io, "  b: $(p.b)")
    println(io, "  ku: $(p.ku)")
    println(io, "  c: $(p.c)")
    println(io, "  v: $(p.v)")
    println(io, "  ω: $(p.ω)")
    println(io, "  n: $(p.n)")
    println(io, "  forcing: $(typeof(p.forcing))")
    print(io, "  currentforcing: $(length(p.currentforcing))-element Vector{Float64}")
end

struct Fuse550Parameters <: FuseParameters
    ϕtens::Float64
    S1max::Float64
    S2max::Float64
    r1::Float64
    r2::Float64
    b::Float64
    ku::Float64
    ki::Float64
    ks::Float64
    c::Float64
    n::Float64
    Acmax::Float64
    ω::Float64
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
    function Fuse550Parameters(;
        ϕtens,
        S1max,
        S2max,
        r1,
        b,
        ku,
        ki,
        ks,
        c,
        n,
        Acmax,
        forcing,
    )
        r2 = 1.0 - r1
        ω = S1max * FUSE_RHO  # ρ from FUSE paper
        new(ϕtens, S1max, S2max, r1, r2, b, ku, ki, ks, c, n, Acmax, ω, forcing, zeros(2))
    end
end

# [explicit]
# [implicit]
struct FuseState <: State
    S::Vector{Float64}
    dS::Vector{Float64}
    Sold::Vector{Float64}
    flows::Vector{Float64}
end

# [explicit]
# [implicit]
function primary(state::FuseState)
    return state.S
end

# [explicit]
# [implicit]
function prepare_state(_::FuseParameters, initial)
    return FuseState(copy(initial), zero(initial), copy(initial), zeros(2))
end

# [implicit]
function apply_update!(state::FuseState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

# [implicit]
function copy_state!(state::FuseState, _)
    copyto!(state.Sold, state.S)
    return
end

# [implicit]
function rewind!(state::FuseState)
    copyto!(state.S, state.Sold)
    return
end

# [explicit]
function scaling_factor(ΔS, S)
    return ((-ΔS > S + 1e-9) ? S / abs(ΔS) : 1.0)
end

# [explicit]
function scale_flows(state::FuseState, fuse::FuseParameters, q1, q2, Δt)
    # Make flows smaller such that they do not exceed available storage.
    (; dS, S) = state
    ΔS1 = Δt * dS[1]
    ΔS2 = Δt * dS[2]
    # Check if steps results in negative storage and compute the necessary factor
    # to proportionally reduce outflows. q12 depends on S1 only.
    # Note: exact mass conservation requires scaling outflows consistently per term by S1, S2.
    # FUSE does so; omitted here for simplicity.
    f1 = scaling_factor(ΔS1, S[1])
    f2 = scaling_factor(ΔS2, S[2])
    return f1 * q1, min(f1, f2) * q2
end

# [explicit]
function explicit_timestep!(state::FuseState, fuse::FuseParameters, Δt)
    (; dS, S, flows) = state
    q1, q2 = waterbalance!(dS, S, fuse)
    q1, q2 = scale_flows(state, fuse, q1, q2, Δt)
    flows[1] += q1 * Δt
    flows[2] += q2 * Δt
    @. S += dS * Δt
    @. S = max(S, 0)  # Avoids negative storage.
    handle_excess!(state, fuse)  # Avoids S > Smax; dispatches on FUSE concept.
    return
end

# [implicit]
function residual!(rhs, state::FuseState, fuse::FuseParameters, Δt)
    waterbalance!(state.dS, state.S, fuse)
    # Newton-Raphson uses the negative residual
    @. rhs = -(state.dS - (state.S - state.Sold) / Δt)
    return
end

# [jacobian]
function jacobian!(J, state::FuseState, fuse::FuseParameters, Δt)
    dwaterbalance!(J, state.S, fuse)
    J[1, 1] -= 1.0 / Δt
    J[2, 2] -= 1.0 / Δt
    return
end

# Wrapped for DifferentialEquations.jl
# [diffeq]
function waterbalance!(du, u, p::DiffEqParams{F}, t) where {F<:FuseParameters}
    dS = @view du[1:2]
    S = @view u[1:2]
    q1, q2 = waterbalance!(dS, S, p.parameters)
    du[end-1] = q1
    du[end] = q2
    return
end

# [diffeq]
function isoutofdomain(u, p::DiffEqParams{F}, t)::Bool where {F<:FuseParameters}
    S = @view u[1:2]
    return any(value < 0 for value in S)
end
