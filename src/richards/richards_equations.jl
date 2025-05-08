# For boundary is nothing

function bottomflux(ψ, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function bottomboundary_jacobian!(ψ, parameters, boundary::Nothing)
    return 0.0
end

function topflux(ψ, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function topboundary_jacobian!(ψ, parameters, boundary::Nothing)
    return 0.0
end

# Precipitation

function topflux(ψ, parameters::RichardsParameters)
    return parameters.currentforcing[1]
end

function topboundary_jacobian!(ψ, parameters::RichardsParameters)
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

function bottomflux(ψ, parameters::RichardsParameters, boundary::HeadBoundary)
    kmean = 0.5 * (parameters.k[1] + boundary.k)
    Δψ = boundary.ψ - ψ[1]
    Δz = 0.5 * parameters.Δz[1]
    return kmean * (Δψ / Δz - 1)
end

function bottomboundary_jacobian!(ψ, parameters::RichardsParameters, boundary::HeadBoundary)
    kmean = 0.5 * (parameters.k[1] + boundary.k)
    Δψ = boundary.ψ - ψ[1]
    dk = 0.5 * parameters.dk[1]
    Δz = 0.5 * parameters.Δz[1]
    return -(kmean / Δz) + dk * (Δψ / Δz - 1)
end

function topflux(ψ, parameters::RichardsParameters, boundary::HeadBoundary)
    kmean = 0.5 * (parameters.k[end] + boundary.k)
    Δψ = boundary.ψ - ψ[end]
    Δz = 0.5 * parameters.Δz[end]
    return kmean * (Δψ / Δz + 1)
end

function topboundary_jacobian!(ψ, parameters::RichardsParameters, boundary::HeadBoundary)
    kmean = 0.5 * (parameters.k[end] + boundary.k)
    Δψ = boundary.ψ - ψ[end]
    dk = 0.5 * parameters.dk[end]
    Δz = 0.5 * parameters.Δz[end]
    return -(kmean / Δz) + dk * (Δψ / Δz + 1)
end

# Free drainage

struct FreeDrainage end

function bottomflux(ψ, parameters::RichardsParameters, boundary::FreeDrainage)
    return -parameters.k[1]
end

function bottomboundary_jacobian!(ψ, parameters::RichardsParameters, boundary::FreeDrainage)
    return -parameters.dk[1]
end

# Full column

function waterbalance!(ψ, parameters::RichardsParameters)
    (; constitutive, Δz, Δz⁻¹, bottomboundary, topboundary, n, ∇q, θ, C, k, k_inter, Δψ) =
        parameters
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
    ∇q[1] += bottomflux(ψ, parameters, bottomboundary)
    ∇q[end] += topflux(ψ, parameters, topboundary)
    ∇q[end] += topflux(ψ, parameters)
    return
end

function explicit_timestep!(state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state.ψ, parameters)
    @. state.ψ += parameters.∇q * Δt
    return
end

function residual!(rhs, state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state.ψ, parameters)
    @. rhs = -(parameters.∇q - parameters.Δz * (parameters.θ - state.θ_old) / Δt)
    return
end

function dwaterbalance!(J, ψ, parameters::RichardsParameters)
    (; constitutive, Δz, Δz⁻¹, bottomboundary, topboundary, n, k_inter, Δψ, dk) = parameters
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
    @. dFᵢdψᵢ = 0.0
    @views @. dFᵢdψᵢ[1:end-1] += -dFᵢ₊₁dψᵢ
    @views @. dFᵢdψᵢ[2:end] += -dFᵢ₋₁dψᵢ

    J.d[1] += bottomboundary_jacobian!(ψ, parameters, bottomboundary)
    J.d[end] += topboundary_jacobian!(ψ, parameters, topboundary)
    J.d[end] += topboundary_jacobian!(ψ, parameters)
    return
end

"""
    Construct the Jacobian matrix for the Richards equation finite difference system.
    Sets coefficients for the tridiagonal matrix representing ∂F/∂ψ from the perspective 
    of cell i, with connections to cells i-1 and i+1.

    Use Δt = ∞ for steady-state simulations.
"""
function jacobian!(J, state, parameters::RichardsParameters, Δt)
    dwaterbalance!(J, state.ψ, parameters)
    @. J.d -= (parameters.C * parameters.Δz) / Δt
    return
end

# Wrapped for DifferentialEquations

function waterbalance!(dψ, ψ, p::DiffEqParams{<:RichardsParameters}, t)
    parameters = p.parameters
    waterbalance!(ψ, parameters)
    @. dψ .= 1.0 / (parameters.Δz * parameters.C) * parameters.∇q
    return
end

function dwaterbalance!(J, ψ, p::DiffEqParams{<:RichardsParameters}, t)
    dwaterbalance!(J, ψ, p.parameters)
    return
end

function isoutofdomain(u, p::DiffEqParams{<:RichardsParameters}, t)
    return false
end
