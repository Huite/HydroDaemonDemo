struct RichardsState <: State
    ψ::Vector{Float}
    ψ_old::Vector{Float}
    θ_old::Vector{Float}
    ∇q::Vector{Float}
end

"""Return the primary state."""
function primary(state::RichardsState)
    return state.ψ
end

function prepare_state(p::RichardsParameters, initial)
    return RichardsState(
        copy(initial),  # ψ
        copy(initial),  # ψ_old,
        zero(initial),
        zero(initial),
    )
end

function apply_update!(state::RichardsState, linearsolver, a)
    @. state.ψ += a * linearsolver.ϕ
    return
end

function copy_state!(state::RichardsState, parameters::RichardsParameters)
    copyto!(state.ψ_old, state.ψ)
    state.θ_old .= moisture_content.(state.ψ_old, parameters.constitutive)
    return
end

function rewind!(state::RichardsState)
    copyto!(state.ψ, state.ψ_old)
    # TODO: this is overwritten anyway in a synchronize?
    # copyto!(state.θ, state.θ_old)
end
