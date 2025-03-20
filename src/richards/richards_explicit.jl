"""
    Synchronize the dependent variables (k, C, θ) based on ψ.
"""
function synchronize!(state::RichardsState, parameters)
    # Conductance
    @. state.k = conductivity(state.ψ, parameters.constitutive)

    k_lower = @view(parameters.k[1:end-1])
    Δz_lower = @view(parameters.Δz[1:end-1])
    k_upper = @view(parameters.k[2:end])
    Δz_upper = @view(parameters.Δz[2:end])

    @. state.k_inter = (k_lower * Δz_lower + k_upper * Δz_upper) / (Δz_lower + Δz_upper)
    @. state.kΔz⁻¹ = state.k_inter / (0.5 * Δz_lower + 0.5 * Δz_upper)
    # Moisture capacity
    @. state.C =
        specific_moisture_capacity(@view(state.ψ[2:end-1]), parameters.constitutive)
end

function explicit_timestep!(state::RichardsState, parameters::RichardsParameters, Δt)
    @show state.ψ
    @show state.k

    @. state.Δψ = @view(state.ψ[2:end]) - @view(state.ψ[1:end-1])

    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])
    C_interior = @view state.C[2:end-1]

    @. state.ψ[2:end-1] +=
        (Δt / C_interior) * (
            (kΔz⁻¹ᵢ₊₁ * Δψᵢ₊₁ - kΔz⁻¹ᵢ₋₁ * Δψᵢ₋₁) +
            (@view(state.k_inter[1:end-1]) - @view(state.k_inter[2:end]))
        )

    qbottom = bottomflux(state, parameters, parameters.bottomboundary)
    qtop =
        topflux(state, parameters, parameters.topboundary) +
        topflux(state, parameters, parameters.forcing)

    # Bottom and top cell: use cell-centered k and interfacial k for k gradient
    state.ψ[1] +=
        (Δt / state.C[1]) *
        (qbottom - (state.kΔz⁻¹[1] * state.Δψ[1]) + (state.k[1] - state.k_inter[1]))
    state.ψ[end] +=
        (Δt / state.C[end]) *
        ((state.kΔz⁻¹[end] * state.Δψ[end]) - qtop + (state.k_inter[end] - state.k[end]))
    return
end
