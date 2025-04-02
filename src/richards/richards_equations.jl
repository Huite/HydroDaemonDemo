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
    synchronize!(state, parameters)
    # Internodal flows
    @. state.∇q = 0.0
    @views @. state.∇q[2:end] -= (state.kΔψΔz⁻¹ + state.k_inter)  # i-1 terms
    @views @. state.∇q[1:end-1] += (state.kΔψΔz⁻¹ + state.k_inter)  # i+1 terms

    # Boundary conditions
    state.∇q[1] += bottomflux(state, parameters, parameters.bottomboundary)
    state.∇q[end] += topflux(state, parameters, parameters.topboundary)
    state.∇q[end] += topflux(state, parameters, parameters.forcing)
    return
end

function explicit_timestep!(state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state, parameters)
    # TODO: check storage component; i.e. storage > 0?
    @. state.ψ += state.∇q * Δt
    return
end

function residual!(
    linearsolver::LinearSolver,
    state::RichardsState,
    parameters::RichardsParameters,
    Δt,
)
    waterbalance!(state, parameters)
    @. linearsolver.rhs = -(state.∇q - parameters.Δz * (state.θ - state.θ_old) / Δt)
    return
end

"""
    Construct the Jacobian matrix for the Richards equation finite difference system.
    Sets coefficients for the tridiagonal matrix representing ∂F/∂ψ from the perspective 
    of cell i, with connections to cells i-1 and i+1.

    Use Δt = ∞ for steady-state simulations.
"""
function jacobian!(
    linearsolver::LinearSolver,
    state::RichardsState,
    parameters::RichardsParameters,
    Δt,
)
    # dFᵢ/dψᵢ₋₁ = (kΔz⁻¹)|ᵢ₋₁ - dk/dψ|ᵢ₋₁ * (ΔψΔz⁻¹ + 1) * Δzᵢ₋₁ / (Δzᵢ₋₁ + Δzᵢ)
    # dFᵢ/dψᵢ = -CΔz/Δt 
    #          - (kΔz⁻¹)|ᵢ₋₁ + dk/dψ|ᵢ * (ΔψΔz⁻¹ + 1)|ᵢ₋₁ * Δzᵢ / (Δzᵢ₋₁ + Δzᵢ)
    #          - (kΔz⁻¹)|ᵢ₊₁ - dk/dψ|ᵢ * (ΔψΔz⁻¹ + 1)|ᵢ₊₁ * Δzᵢ / (Δzᵢ₊₁ + Δzᵢ)
    # dFᵢ/dψᵢ₊₁ = (kΔz⁻¹)|ᵢ₊₁ + dk/dψ|ᵢ₊₁ * (ΔψΔz⁻¹ + 1) * Δzᵢ₊₁ / (Δzᵢ₊₁ + Δzᵢ)

    J = linearsolver.J
    n = linearsolver.n
    dFᵢdψᵢ = J.d  # derivatives of F₁, ... Fₙ with respect to ψ₁, ... ψₙ
    dFᵢ₊₁dψᵢ = J.dl  # derivatives of F₂, ... Fₙ with respect to ψ₁, ... ψₙ₋₁
    dFᵢ₋₁dψᵢ = J.du  # derivatives of F₁, ... Fₙ₋₁ with respect to ψ₂, ... ψₙ

    # TODO: Probably swap these variable names
    dkᵢ₋₁ = @view(state.dk[1:end-1])
    dkᵢ₊₁ = @view(state.dk[2:end])
    Δzᵢ₋₁ = @view(parameters.Δz[1:end-1])
    Δzᵢ₊₁ = @view(parameters.Δz[2:end])

    # First compute the off-diagonal terms
    @. dFᵢ₊₁dψᵢ = state.kΔz⁻¹ - dkᵢ₋₁ * state.ΔψΔz⁻¹ * Δzᵢ₋₁ / (Δzᵢ₋₁ + Δzᵢ₊₁)
    @. dFᵢ₋₁dψᵢ = state.kΔz⁻¹ + dkᵢ₊₁ * state.ΔψΔz⁻¹ * Δzᵢ₊₁ / (Δzᵢ₊₁ + Δzᵢ₋₁)

    # Then compute the diagonal term
    @. dFᵢdψᵢ = -(state.C * parameters.Δz) / Δt
    @views @. dFᵢdψᵢ[1:end-1] += -dFᵢ₊₁dψᵢ
    @views @. dFᵢdψᵢ[2:end] += -dFᵢ₋₁dψᵢ

    J.d[1] += bottomboundary_jacobian!(state, parameters, parameters.bottomboundary)
    J.d[end] += topboundary_jacobian!(state, parameters, parameters.topboundary)
    J.d[end] += topboundary_jacobian!(state, parameters, parameters.forcing)
    return
end

function isoutofdomain(
    u::Vector{T},
    p::DiffEqParams{RP,RichardsState},
    t::Real,
) where {T,RP}
    return false
end
