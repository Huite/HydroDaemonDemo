# For boundary is nothing

function bottomflux(ψ, parameters::AbstractRichards, boundary::Nothing)
    return 0.0
end

function bottomboundary_jacobian(ψ, parameters, boundary::Nothing)
    return 0.0
end

function topflux(ψ, parameters::AbstractRichards, boundary::Nothing)
    return 0.0
end

function topboundary_jacobian(ψ, parameters, boundary::Nothing)
    return 0.0
end

# Precipitation

function forcingflux(ψ, parameters::AbstractRichards)
    return parameters.currentforcing[1]
end

function forcing_jacobian(ψ, parameters::AbstractRichards)
    return 0.0
end

# Store k value since it never changes
struct HeadBoundary
    ψ::Float64
    k::Float64
end

function HeadBoundary(ψ, constitutive::ConstitutiveRelationships)
    return HeadBoundary(ψ, conductivity(ψ, constitutive))
end

function bottomflux(ψ, parameters::AbstractRichards, boundary::HeadBoundary)
    kmean = 0.5 * (conductivity(ψ[1], parameters.constitutive[1]) + boundary.k)
    Δψ = boundary.ψ - ψ[1]
    Δz = 0.5 * parameters.Δz
    return kmean * (Δψ / Δz - 1)
end

function bottomboundary_jacobian(ψ, parameters::AbstractRichards, boundary::HeadBoundary)
    kmean = 0.5 * (conductivity(ψ[1], parameters.constitutive[1]) + boundary.k)
    Δψ = boundary.ψ - ψ[1]
    dk = 0.5 * dconductivity(ψ[1], parameters.constitutive[1])
    Δz = 0.5 * parameters.Δz
    return -(kmean / Δz) + dk * (Δψ / Δz - 1)
end

function topflux(ψ, parameters::AbstractRichards, boundary::HeadBoundary)
    kmean = 0.5 * (conductivity(ψ[end], parameters.constitutive[end]) + boundary.k)
    Δψ = boundary.ψ - ψ[end]
    Δz = 0.5 * parameters.Δz
    return kmean * (Δψ / Δz + 1)
end

function topboundary_jacobian(ψ, parameters::AbstractRichards, boundary::HeadBoundary)
    kmean = 0.5 * (conductivity(ψ[end], parameters.constitutive[end]) + boundary.k)
    Δψ = boundary.ψ - ψ[end]
    dk = 0.5 * dconductivity(ψ[end], parameters.constitutive[end])
    Δz = 0.5 * parameters.Δz
    return -(kmean / Δz) + dk * (Δψ / Δz + 1)
end

# Free drainage

struct FreeDrainage end

function bottomflux(ψ, parameters::AbstractRichards, boundary::FreeDrainage)
    return -conductivity(ψ[1], parameters.constitutive[1])
end

function bottomboundary_jacobian(ψ, parameters::AbstractRichards, boundary::FreeDrainage)
    return -dconductivity(ψ[1], parameters.constitutive[1])
end

# Full column

function waterbalance!(∇q, ψ, parameters::AbstractRichards)
    (; constitutive, Δz, bottomboundary, topboundary, n) = parameters
    @. ∇q = 0.0
    Δz⁻¹ = 1.0 / Δz

    # Internodal flows
    k_lower = conductivity(ψ[1], constitutive[1])
    for i = 1:(n-1)
        upper = i + 1
        k_upper = conductivity(ψ[upper], constitutive[upper])
        k_inter = 0.5 * (k_lower + k_upper)
        Δψ = ψ[upper] - ψ[i]
        q = k_inter * (Δψ * Δz⁻¹ + 1)
        ∇q[i] += q
        ∇q[upper] -= q
        k_lower = k_upper
    end

    # Boundary conditions
    qbot = bottomflux(ψ, parameters, bottomboundary)
    qtop = topflux(ψ, parameters, topboundary) + forcingflux(ψ, parameters)
    ∇q[1] += qbot
    ∇q[end] += qtop
    return qbot, qtop
end

function explicit_timestep!(state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state.∇q, state.ψ, parameters)
    @. state.ψ += state.∇q * Δt
    return
end

# For handwritten Newton formulation.

function residual!(rhs, state::RichardsState, parameters::RichardsParameters, Δt)
    waterbalance!(state.∇q, state.ψ, parameters)
    Δz = parameters.Δz
    for i = 1:parameters.n
        θ = moisture_content(state.ψ[i], parameters.constitutive[i])
        rhs[i] = -(state.∇q[i] - Δz * (θ - state.θ_old[i]) / Δt)
    end
    return
end

function dwaterbalance!(J, ψ, parameters::RichardsParameters)
    (; constitutive, Δz, bottomboundary, topboundary, n) = parameters

    dFᵢdψᵢ = J.d  # derivatives of F₁, ... Fₙ with respect to ψ₁, ... ψₙ
    dFᵢ₊₁dψᵢ = J.dl  # derivatives of F₂, ... Fₙ with respect to ψ₁, ... ψₙ₋₁
    dFᵢ₋₁dψᵢ = J.du  # derivatives of F₁, ... Fₙ₋₁ with respect to ψ₂, ... ψₙ
    Δz⁻¹ = 1.0 / Δz

    # First compute the off-diagonal terms -- relating to the internodal flows.
    k_lower = conductivity(ψ[1], constitutive[1])
    dk_lower = dconductivity(ψ[1], constitutive[1])
    for i = 1:(n-1)
        upper = i + 1
        k_upper = conductivity(ψ[upper], constitutive[upper])
        dk_upper = dconductivity(ψ[upper], constitutive[upper])
        k_inter = 0.5 * (k_lower + k_upper)
        Δψ = ψ[upper] - ψ[i]
        dFᵢ₊₁dψᵢ[i] = (k_inter * Δz⁻¹) - dk_lower * (Δψ * Δz⁻¹) * 0.5
        dFᵢ₋₁dψᵢ[i] = (k_inter * Δz⁻¹) + dk_lower * (Δψ * Δz⁻¹) * 0.5
        k_lower = k_upper
        dk_lower = dk_upper
    end

    # Then compute the diagonal term
    @. dFᵢdψᵢ = 0.0
    @views @. dFᵢdψᵢ[1:end-1] += -dFᵢ₊₁dψᵢ
    @views @. dFᵢdψᵢ[2:end] += -dFᵢ₋₁dψᵢ

    J.d[1] += bottomboundary_jacobian(ψ, parameters, bottomboundary)
    J.d[end] += topboundary_jacobian(ψ, parameters, topboundary)
    J.d[end] += forcing_jacobian(ψ, parameters)
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
    Δz = parameters.Δz
    for i = 1:parameters.n
        C = specific_moisture_capacity(state.ψ[i], parameters.constitutive[i])
        Sa = aqueous_saturation(state.ψ[i], parameters.constitutive[i])
        Ss = parameters.constitutive[i].Ss
        J.d[i] -= (Δz * (C + Sa * Ss)) / Δt
    end
    return
end

# Wrapped for DifferentialEquations
# This is the "ψ-based" Richards equation.

function waterbalance!(du, u, p::DiffEqParams{<:RichardsParameters}, t)
    @views dψ = du[1:end-2]
    @views ψ = u[1:end-2]
    parameters = p.parameters
    qbot, qtop = waterbalance!(dψ, ψ, parameters)
    Δz = parameters.Δz
    for i = 1:parameters.n
        C = specific_moisture_capacity(ψ[i], parameters.constitutive[i])
        Sa = aqueous_saturation(ψ[i], parameters.constitutive[i])
        Ss = parameters.constitutive[i].Ss
        dψ[i] *= 1.0 / (Δz * (C + Sa * Ss))
    end
    du[end-1] = qbot * 1e-9
    du[end] = qtop * 1e-9
    return
end

function isoutofdomain(u, p::DiffEqParams{<:AbstractRichards}, t)
    return false
end

function waterbalance_dae!(du, u, parameters::RichardsParametersDAE)
    n = parameters.n
    dψ = @view du[1:n]  # Acts as ∇q first
    ψ = @view u[1:n]
    dθ = @view du[n+1:end-2]
    θ = @view u[n+1:end-2]

    qbot, qtop = waterbalance!(dψ, ψ, parameters)
    Δz = parameters.Δz
    for i = 1:parameters.n
        # Head-based Richards equation
        C = specific_moisture_capacity(ψ[i], parameters.constitutive[i])
        Sa = θ[i] / parameters.constitutive[i].θs
        Ss = parameters.constitutive[i].Ss
        dψ[i] /= (Δz * (C + Sa * Ss))
        # Algebraic constraint
        dθ[i] = θ[i] - moisture_content(ψ[i], parameters.constitutive[i])
    end
    du[end-1] = qbot * 1e-9
    du[end] = qtop * 1e-9
    return
end

function waterbalance!(du, u, p::DiffEqParams{<:RichardsParametersDAE}, t)
    waterbalance_dae!(du, u, p.parameters)
    return
end
