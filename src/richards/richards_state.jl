struct RichardsState
    ψ::Vector{Float}
    ψ_old::Vector{Float}
    θ_old::Vector{Float}
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
    )
end

function apply_update!(state::RichardsState, linearsolver, a)
    @. state.ψ += a * linearsolver.ϕ
    return
end

function copy_state!(state::RichardsState)
    copyto!(state.ψ_old, state.ψ)
    copyto!(state.θ_old, state.θ)
end

function rewind!(state::RichardsState)
    copyto!(state.ψ, state.ψ_old)
    # TODO: this is overwritten anyway in a synchronize?
    # copyto!(state.θ, state.θ_old)
end

function compute_timestep_size(cfl::CFLTimeStepper, state::RichardsState, parameters, Δt)
    # Maximum diffusivity-based Δt
    max_diffusivity = maximum(state.k ./ state.C)
    min_Δz = minimum(parameters.Δz)
    Δt_diffusive = cfl.target * min_Δz^2 / (2 * max_diffusivity)

    # Maximum gravity-based Δt (for stability of the gravity term)
    max_k_gradient = maximum(abs.(diff(state.k)))
    Δt_gravity = cfl.target * minimum(state.C) * min_Δz / max_k_gradient

    # Account for boundary conditions
    if parameters.bottomboundary isa HeadBoundary
        bottom_ψ = parameters.bottomboundary.ψ
        bottom_gradient = abs(state.ψ[1] - bottom_ψ) / (0.5 * parameters.Δz[1])
        max_gradient = max(bottom_gradient, max_k_gradient)
    end

    if parameters.topboundary isa HeadBoundary
        top_ψ = parameters.topboundary.ψ
        top_gradient = abs(state.ψ[end] - top_ψ) / (0.5 * parameters.Δz[end])
        max_gradient = max(top_gradient, max_k_gradient)
    end

    # Add boundary gradient constraint
    Δt_boundary = cfl.target * minimum(state.C) * min_Δz^2 / max_gradient

    # Take the most restrictive timestep
    Δt_restricted = min(Δt_diffusive, Δt_gravity, Δt_boundary)

    # Add a safety factor and enforce a minimum/maximum
    Δt_cfl = max(min(0.9 * Δt_restricted, cfl.Δt0), cfl.minstep)

    return min(Δt_cfl, Δt)
end
