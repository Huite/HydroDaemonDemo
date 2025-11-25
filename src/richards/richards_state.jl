struct RichardsState <: State
    ψ::Vector{Float64}
    ψ_old::Vector{Float64}
    θ_old::Vector{Float64}
    ∇q::Vector{Float64}
    flows::Vector{Float64}
end

# [explicit]
# [implicit]
"""Return the primary state."""
function primary(state::RichardsState)
    return state.ψ
end

# [explicit]
# [implicit]
function compute_savedflows!(state::RichardsState, parameters::RichardsParameters, Δt)
    state.flows[1] += Δt * bottomflux(state.ψ, parameters, parameters.bottomboundary)
    state.flows[2] +=
        Δt * (
            topflux(state.ψ, parameters, parameters.topboundary) +
            forcingflux(state.ψ, parameters)
        )
    return
end

# [implicit]
function prepare_state(p::RichardsParameters, initial)
    return RichardsState(
        copy(initial),  # ψ
        copy(initial),  # ψ_old,
        zero(initial),
        zero(initial),
        zeros(2),  # qbottom, qtop
    )
end

# [nonlinear_solve]
function apply_update!(state::RichardsState, linearsolver, a)
    @. state.ψ += a * linearsolver.ϕ
    return
end

# [implicit]
function copy_state!(state::RichardsState, parameters::RichardsParameters)
    copyto!(state.ψ_old, state.ψ)
    state.θ_old .= moisture_content.(state.ψ_old, parameters.constitutive)
    return
end

# [implicit]
function rewind!(state::RichardsState)
    copyto!(state.ψ, state.ψ_old)
end
