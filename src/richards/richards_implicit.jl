"""
    Synchronize the dependent variables (k, C, θ) based on ψ.
"""
function synchronize!(state::RichardsState, parameters)
    # Conductance
    @. state.k = conductivity(state.ψ, parameters)
    @. state.dk = dconductivity(state.ψ, parameters)
    @. state.k_inter = 0.5 * (@view(state.k[1:end-1]) + @view(state.k[2:end]))
    @. state.kΔz⁻¹ = state.k_inter / parameters.Δz

    # Moisture capacity
    @. state.C = specific_moisture_capacity(@view(state.ψ[2:end-1]), parameters)
    @. state.dS = state.C * parameters.Δz

    # Moisture content
    @. state.θ = moisture_content(state.ψ[2:end-1], parameters)
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
    Formulate residual and set rhs.

    Use Δt = ∞ for steady-state.
"""
function residual!(
    linearsolver::LinearSolver,
    state::RichardsState,
    parameters::RichardsParameters,
    Δt,
)
    r = linearsolver.r

    @. state.Δψ = @view(state.ψ[2:end]) - @view(state.ψ[1:end-1])
    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])

    # storage
    @. r = state.Δz * (state.θ_old - state.θ) / Δt
    # flow
    @. r += kΔz⁻¹ᵢ₊₁ * Δψᵢ₊₁ - kΔz⁻¹ᵢ₋₁ * Δψᵢ₋₁
    # gravity
    @. r += (@view(state.k_inter[1:end-1]) - @view(state.k_inter[2:end]))

    topboundary_residual!(r, state, parameters.topboundary)
    bottomboundary_residual!(r, state, parameters.bottomboundary)
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
    J = linearsolver.J

    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    Δz = parameters.Δz

    # Calculate tridiagonal Jacobian matrix coefficients
    @. J.dl = -kΔz⁻¹ᵢ₋₁[2:end]
    @. J.d = state.dS / Δt + kΔz⁻¹ᵢ₋₁ + kΔz⁻¹ᵢ₊₁
    @. J.du = -kΔz⁻¹ᵢ₊₁[1:end-1]

    # Add linearized conductivity terms
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    dkᵢ₋₁ = @view(state.dk[1:end-2])
    dkᵢ₊₁ = @view(state.dk[3:end])

    Δψᵢ₋₁l = @view(Δψᵢ₋₁[2:end])
    Δψᵢ₊₁u = @view(Δψᵢ₊₁[1:end-1])
    dkᵢ₋₁l = @view(dkᵢ₋₁[2:end])
    dkᵢ₊₁u = @view(dkᵢ₊₁[1:end-1])

    @. J.dl += -dkᵢ₋₁l * (Δψᵢ₋₁l / Δz + 1.0)
    @. J.d += -dkᵢ₋₁ * (Δψᵢ₋₁ / Δz + 1.0) + dkᵢ₊₁ * (Δψᵢ₊₁ / Δz + 1.0)
    @. J.du += dkᵢ₊₁u * (Δψᵢ₊₁u / Δz + 1.0)

    topboundary_jacobian!(J, state, parameters.topboundary)
    bottomboundary_jacobian!(J, state, parameters.bottomboundary)
    return
end
