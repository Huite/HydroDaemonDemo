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
    n::Int
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
    function Fuse070Parameters(; ϕtens, S1max, b, ku, c, v, forcing)
        ω = S1max * FUSE_RHO  # ρ from FUSE paper
        nstate = 2
        new(ϕtens, S1max, b, ku, c, v, ω, nstate, forcing, zeros(2))
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

function compute_savedflows!(state::FuseState, parameters::FuseParameters, Δt)
    # Compute flows based on the current solution.
    q1, q2 = waterbalance!(state.dS, state.S, parameters)
    state.flows[1] += Δt * q1
    state.flows[2] += Δt * q2
    return
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

function scaling_factor(ΔS, S)
    return ((-ΔS > S + 1e-9) ? S / abs(ΔS) : 1.0)
end

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
    dS = @view du[1:2]
    S = @view u[1:2]
    q1, q2 = waterbalance!(dS, S, p.parameters)
    du[end-1] = q1
    du[end] = q2
    return
end

function isoutofdomain(u, p::DiffEqParams{F}, t)::Bool where {F<:FuseParameters}
    S = @view u[1:2]
    return any(value < 0 for value in S)
end
