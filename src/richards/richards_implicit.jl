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
    # Fᵢ = - (kΔz⁻¹Δψ)|ᵢ₋₁ - kΔz⁻¹|ᵢ₋₁ + (kΔz⁻¹ Δψ)|ᵢ₊₁ + kΔz⁻¹|ᵢ₊₁ + qⱼ - Δzᵢ (θᵢᵗ⁺¹ - θᵢᵗ) / Δt 
    F = linearsolver.rhs

    @. F = -(parameters.Δz * (state.θ - state.θ_old) / Δt)  # i terms
    @. F[2:end] += -state.kΔψΔz⁻¹  - state.k_inter  # i-1 terms
    @. F[1:end - 1] += state.kΔψΔz⁻¹ + state.k_inter  # i+1 terms

    topboundary_residual!(F, state, parameters, parameters.forcing)
    topboundary_residual!(F, state, parameters, parameters.topboundary)
    bottomboundary_residual!(F, state, parameters, parameters.bottomboundary)

    # Flip the sign, since the Newton-Raphson scheme uses the negative residual.
    @. linearsolver.rhs = -F
    return
end

"""
    Formulate and set coefficients in the matrix.

    Use Δt = ∞ for steady-state.
"""
function jacobian!(
    linearsolver::LinearSolver,
    state::RichardsState,
    parameters::RichardsParameters,
    Δt,
)
    # dFᵢ/dψᵢ₋₁ = (kΔz⁻¹)|ᵢ₋₁ - dk/dψ|ᵢ₋₁ * (ΔψΔz⁻¹ + 1) * Δzᵢ₋₁ / (Δzᵢ₋₁ + Δzᵢ)
    # dFᵢ/dψᵢ = -CΔz/Δt
    # dFᵢ/dψᵢ₊₁ =  (kΔz⁻¹)|ᵢ₊₁ - dk/dψ|ᵢ₊₁ * (ΔψΔz⁻¹ + 1) * Δzᵢ₊₁ / (Δzᵢ₊₁ + Δzᵢ)

    J = linearsolver.J
    dFᵢdψᵢ = J.d
    dFᵢdψᵢ₋₁ = J.dl
    dFᵢdψᵢ₊₁ = J.du

    dkᵢ₋₁ = @view(state.dk[1:end-1])
    dkᵢ₊₁ = @view(state.dk[2:end])
    Δzᵢ₋₁ = @view(parameters.Δz[1:end-1])
    Δzᵢ₊₁ = @view(parameters.Δz[2:end])

    @. dFᵢdψᵢ₋₁ = state.kΔz⁻¹ - dkᵢ₋₁  * state.ΔψΔz⁻¹ * Δzᵢ₋₁ / (Δzᵢ₋₁ + Δzᵢ₊₁)

    # terms for i
    @. dFᵢdψᵢ = -(state.C * parameters.Δz) / Δt
    # terms for i-1
    @. dFᵢdψᵢ[1:end-1] += -state.kΔz⁻¹ - dkᵢ₋₁ * state.ΔψΔz⁻¹ * Δzᵢ₋₁ / (Δzᵢ₋₁ + Δzᵢ₊₁)
    # terms for i+1
    @. dFᵢdψᵢ[2:end] += -state.kΔz⁻¹ + dkᵢ₊₁ * state.ΔψΔz⁻¹ * Δzᵢ₊₁ / (Δzᵢ₊₁ + Δzᵢ₋₁)

    @. dFᵢdψᵢ₊₁ = state.kΔz⁻¹ + dkᵢ₊₁ * state.ΔψΔz⁻¹ * Δzᵢ₊₁ / (Δzᵢ₊₁ + Δzᵢ₋₁)

    topboundary_jacobian!(J, state, parameters, parameters.forcing)
    topboundary_jacobian!(J, state, parameters, parameters.topboundary)
    bottomboundary_jacobian!(J, state, parameters, parameters.bottomboundary)
    return
end
