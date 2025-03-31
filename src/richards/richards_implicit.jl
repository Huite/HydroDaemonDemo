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
    Formulate residual and set rhs.

    Use Δt = ∞ for steady-state.
"""
function residual!(
    linearsolver::LinearSolver,
    state::RichardsState,
    parameters::RichardsParameters,
    Δt,
)
    synchronize!(state, parameters)

    # Fᵢ = - (kΔz⁻¹Δψ)|ᵢ₋₁ - kΔz⁻¹|ᵢ₋₁ + (kΔz⁻¹ Δψ)|ᵢ₊₁ + kΔz⁻¹|ᵢ₊₁ + qᵢ - Δzᵢ (θᵢᵗ⁺¹ - θᵢᵗ) / Δt 
    F = linearsolver.rhs

    @. F = -(parameters.Δz * (state.θ - state.θ_old) / Δt)  # i terms
    @. F[2:end] += -state.kΔψΔz⁻¹ - state.k_inter  # i-1 terms
    @. F[1:end-1] += state.kΔψΔz⁻¹ + state.k_inter  # i+1 terms

    topboundary_residual!(F, state, parameters, parameters.forcing)
    topboundary_residual!(F, state, parameters, parameters.topboundary)
    bottomboundary_residual!(F, state, parameters, parameters.bottomboundary)

    # Flip the sign, since the Newton-Raphson scheme uses the negative residual.
    @. linearsolver.rhs = -F
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
    @. dFᵢdψᵢ[1:end-1] += -dFᵢ₊₁dψᵢ
    @. dFᵢdψᵢ[2:end] += -dFᵢ₋₁dψᵢ

    topboundary_jacobian!(J, state, parameters, parameters.forcing)
    topboundary_jacobian!(J, state, parameters, parameters.topboundary)
    bottomboundary_jacobian!(J, state, parameters, parameters.bottomboundary)
    return
end
