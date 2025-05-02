"""
This struct holds the mutable members of the Richards 1D simulation.
"""

struct RichardsState <: State
    ψ::Vector{Float}
    ∇q::Vector{Float}  # ∇q/dt
    θ::Vector{Float}
    ψ_old::Vector{Float}
    θ_old::Vector{Float}
    # specific moisture capacity
    C::Vector{Float}
    # conductivity
    k::Vector{Float}
    # Internodal data, all size n - 1!
    k_inter::Vector{Float}
    Δψ::Vector{Float}  # ψᵢ - ψᵢ₋₁
    kΔz⁻¹::Vector{Float}  # k/Δz
    ΔψΔz⁻¹::Vector{Float}  # Δψ/Δz
    kΔψΔz⁻¹::Vector{Float}  # kΔψ/Δz
    # Newton-Raphson work array, size n
    dk::Vector{Float}  # dk/dψ
    # Forcing, size 2
    forcing::Vector{Float}
end

"""Return the primary state."""
function primary(state::RichardsState)
    return state.ψ
end

function prepare_state(p::RichardsParameters, initial)
    n = length(p.constitutive)
    return RichardsState(
        copy(initial),  # ψ
        zeros(n),  # ∇q
        zeros(n),  # θ
        zeros(n),  # ψ_old
        zeros(n),  # θ_old
        zeros(n),  # C
        zeros(n),  # k
        zeros(n - 1),  # k_inter
        zeros(n - 1),  # Δψ
        zeros(n - 1),  # kΔz⁻¹
        zeros(n - 1),  # ΔψΔz⁻¹
        zeros(n - 1),  # kΔψΔz⁻¹
        zeros(n),  # dk
        zeros(2),
    )
end

function force!(state::RichardsState, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
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

"""
    Synchronize the dependent variables (k, C, θ) based on ψ.
"""
function synchronize!(state::RichardsState, parameters)
    @. state.k = conductivity(state.ψ, parameters.constitutive)
    @. state.dk = dconductivity(state.ψ, parameters.constitutive)

    k_lower = @view(state.k[1:end-1])
    Δz_lower = @view(parameters.Δz[1:end-1])
    k_upper = @view(state.k[2:end])
    Δz_upper = @view(parameters.Δz[2:end])
    ψ_upper = @view state.ψ[2:end]
    ψ_lower = @view state.ψ[1:end-1]

    @. state.Δψ = ψ_upper - ψ_lower
    @. state.k_inter = (k_lower * Δz_lower + k_upper * Δz_upper) / (Δz_lower + Δz_upper)
    @. state.kΔz⁻¹ = state.k_inter / (0.5 * Δz_lower + 0.5 * Δz_upper)
    @. state.kΔψΔz⁻¹ = state.Δψ * state.kΔz⁻¹
    @. state.ΔψΔz⁻¹ = state.Δψ / (0.5 * Δz_lower + 0.5 * Δz_upper) + 1.0

    # Moisture capacity
    @. state.C = specific_moisture_capacity(state.ψ, parameters.constitutive)

    # Moisture content
    @. state.θ = moisture_content(state.ψ, parameters.constitutive)

    return
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
