abstract type FuseParameters <: Parameters end

const FUSE_RHO = 0.01

struct Fuse070Parameters <: FuseParameters
    ϕtens::Float64
    S1max::Float64
    b::Float64
    ku::Float64
    c::Float64
    v::Float64
    ω::Float64
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
    function Fuse070Parameters(; ϕtens, S1max, b, ku, c, v, forcing)
        ω = S1max * FUSE_RHO  # ρ from FUSE paper
        new(ϕtens, S1max, b, ku, c, v, ω, forcing, zeros(2))
    end
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

struct FuseState <: State
    S::Vector{Float64}
    dS::Vector{Float64}
    Sold::Vector{Float64}
    flows::Vector{Float64}
end

function primary(state::FuseState)
    return state.S
end

function prepare_state(_::FuseParameters, initial)
    return FuseState(copy(initial), zero(initial), copy(initial), zeros(2))
end

function apply_update!(state::FuseState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

function copy_state!(state::FuseState, _)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::FuseState)
    copyto!(state.S, state.Sold)
    return
end

function explicit_timestep!(state::FuseState, fuse::FuseParameters, Δt)
    (; dS, S) = state
    waterbalance!(dS, S, fuse)
    @. state.S += state.dS * Δt
    @. state.S = max(state.S, 0)
    return
end

function residual!(rhs, state::FuseState, fuse::FuseParameters, Δt)
    waterbalance!(state.dS, state.S, fuse)
    # Newton-Raphson uses the negative residual
    @. rhs = -(state.dS - (state.S - state.Sold) / Δt)
    return
end
function jacobian!(J, state::FuseState, fuse::FuseParameters, Δt)
    dwaterbalance!(J, state.S, fuse)
    J[1, 1] -= 1.0 / Δt
    J[2, 2] -= 1.0 / Δt
    return
end

# Wrapped for DifferentialEquations.jl

function waterbalance!(du, u, p::DiffEqParams{F}, t) where {F<:FuseParameters}
    S = @view u[1:2]
    e, q = waterbalance!(du, u, p.parameters)
    du[end-1] = e
    du[end] = q
    return
end

function isoutofdomain(u, p::DiffEqParams{F}, t)::Bool where {F<:FuseParameters}
    S = @view u[1:2]
    return any(value < 0 for value in S)
end
