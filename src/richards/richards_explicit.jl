

struct RicardsExplicitState <: RichardsState
    ψ::Vector{Float}
    θ::Vector{Float}
    # specific moisture capacity
    C::Vector{Float}
    # conductivity
    k::Vector{Float}
    k_inter::Vector{Float}
    kΔz⁻¹::Vector{Float}
    Δψ::Vector{Float}  # Δψ/Δz
    forcing::Vector{Float}
end

"""
    Synchronize the dependent variables (k, C, θ) based on ψ.
"""
function synchronize!(state::RichardsExplicitState, parameters)
    # Conductance
    @. state.k = conductivity(state.ψ, parameters)
    @. state.k_inter = 0.5 * (@view(state.k[1:end-1]) + @view(state.k[2:end]))
    @. state.kΔz⁻¹ = state.k_inter / parameters.Δz
    # Moisture capacity
    @. state.C = specific_moisture_capacity(@view(state.ψ[2:end-1]), parameters)
end

function timestep(column, Δt)
    state = column.state
    synchronize!(state, column.parameters)

    @. state.Δψ = @view(state.ψ[2:end]) - @view(state.ψ[1:end-1])

    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])

    @. state.ψ +=
        Δt / state.C * (kΔz⁻¹ᵢ₊₁ * Δψᵢ₊₁ - kΔz⁻¹ᵢ₋₁ * Δψᵢ₋₁) +
        (@view(state.k_inter[1:end-1]) - @view(state.k_inter[2:end]))
end
