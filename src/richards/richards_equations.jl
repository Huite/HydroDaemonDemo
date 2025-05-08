# For boundary is nothing

function bottomflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function bottomboundary_jacobian!(state, parameters, boundary::Nothing)
    return 0.0
end

function topflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function topboundary_jacobian!(state, parameters, boundary::Nothing)
    return 0.0
end

# Precipitation

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    return state.forcing[1]
end

function topboundary_jacobian!(
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    return 0.0
end

# Store k value since it never changes
struct HeadBoundary
    ψ::Float
    k::Float
end

function HeadBoundary(ψ, constitutive::ConstitutiveRelationships)
    return HeadBoundary(ψ, conductivity(ψ, constitutive))
end

function bottomflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[1] + boundary.k)
    Δψ = boundary.ψ - state.ψ[1]
    Δz = 0.5 * parameters.Δz[1]
    return kmean * (Δψ / Δz - 1)
end

function bottomboundary_jacobian!(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[1] + boundary.k)
    Δψ = boundary.ψ - state.ψ[1]
    dk = 0.5 * state.dk[1]
    Δz = 0.5 * parameters.Δz[1]
    return -(kmean / Δz) + dk * (Δψ / Δz - 1)
end

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[end] + boundary.k)
    Δψ = boundary.ψ - state.ψ[end]
    Δz = 0.5 * parameters.Δz[end]
    return kmean * (Δψ / Δz + 1)
end

function topboundary_jacobian!(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[end] + boundary.k)
    Δψ = boundary.ψ - state.ψ[end]
    dk = 0.5 * state.dk[end]
    Δz = 0.5 * parameters.Δz[end]
    return -(kmean / Δz) + dk * (Δψ / Δz + 1)
end

# Free drainage

struct FreeDrainage end

function bottomflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    return -state.k[1]
end

function bottomboundary_jacobian!(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    return -state.dk[1]
end

# Full column

function waterbalance!(state::RichardsState, parameters::RichardsParameters)
    (; ψ, ∇q, θ, C, k, k_inter, Δψ, forcing) = state
    (; constitutive, Δz, Δz⁻¹, forcing, bottomboundary, topboundary, n) = parameters
    @. k = conductivity(ψ, constitutive)
    @. C = specific_moisture_capacity(ψ, constitutive)
    @. θ = moisture_content(ψ, constitutive)

    lower = (1:n-1)
    upper = (2:n)
    @. Δψ = ψ[upper] - ψ[lower]
    @. k_inter = (k[lower] * Δz[lower] + k[upper] * Δz[upper]) / (Δz[lower] + Δz[upper])

    # Internodal flows
    @. ∇q = 0.0
    @views @. ∇q[2:end] -= k_inter * (Δψ * Δz⁻¹ + 1)  # i-1 terms
    @views @. ∇q[1:end-1] += k_inter * (Δψ * Δz⁻¹ + 1)  # i+1 terms

    # Boundary conditions
    ∇q[1] += bottomflux(state, parameters, bottomboundary)
    ∇q[end] += topflux(state, parameters, topboundary)
    ∇q[end] += topflux(state, parameters, forcing)
    return
end

function explicit_timestep!(state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state, parameters)
    # TODO: check storage component; i.e. storage > 0?
    @. state.ψ += state.∇q * Δt
    return
end

function residual!(rhs, state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state, parameters)
    @. rhs = -(state.∇q - parameters.Δz * (state.θ - state.θ_old) / Δt)
    return
end

"""
    Construct the Jacobian matrix for the Richards equation finite difference system.
    Sets coefficients for the tridiagonal matrix representing ∂F/∂ψ from the perspective 
    of cell i, with connections to cells i-1 and i+1.

    Use Δt = ∞ for steady-state simulations.
"""
function jacobian!(J, state::RichardsState, parameters::RichardsParameters, Δt)
    (; ψ, C, k_inter, Δψ, dk, forcing) = state
    (; constitutive, Δz, Δz⁻¹, forcing, bottomboundary, topboundary, n) = parameters
    @. dk = dconductivity(ψ, constitutive)

    dFᵢdψᵢ = J.d  # derivatives of F₁, ... Fₙ with respect to ψ₁, ... ψₙ
    dFᵢ₊₁dψᵢ = J.dl  # derivatives of F₂, ... Fₙ with respect to ψ₁, ... ψₙ₋₁
    dFᵢ₋₁dψᵢ = J.du  # derivatives of F₁, ... Fₙ₋₁ with respect to ψ₂, ... ψₙ

    lower = (1:n-1)
    upper = (2:n)
    # First compute the off-diagonal terms
    @. dFᵢ₊₁dψᵢ =
        (k_inter * Δz⁻¹) - dk[lower] * (Δψ * Δz⁻¹) * Δz[lower] / (Δz[lower] + Δz[upper])
    @. dFᵢ₋₁dψᵢ =
        (k_inter * Δz⁻¹) + dk[lower] * (Δψ * Δz⁻¹) * Δz[upper] / (Δz[upper] + Δz[lower])

    # Then compute the diagonal term
    @. dFᵢdψᵢ = -(C * Δz) / Δt
    @views @. dFᵢdψᵢ[1:end-1] += -dFᵢ₊₁dψᵢ
    @views @. dFᵢdψᵢ[2:end] += -dFᵢ₋₁dψᵢ

    J.d[1] += bottomboundary_jacobian!(state, parameters, bottomboundary)
    J.d[end] += topboundary_jacobian!(state, parameters, topboundary)
    J.d[end] += topboundary_jacobian!(state, parameters, forcing)
    return
end

function isoutofdomain(
    u::Vector{T},
    p::DiffEqParams{RP,RichardsState},
    t::Real,
) where {T,RP}
    return false
end

function righthandside!(du, state::RichardsState, parameters::RichardsParameters)
    @. du = state.∇q / (parameters.Δz * state.C)
    return
end
