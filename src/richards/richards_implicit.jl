"""
This struct holds the mutable members of the Richards 1D simulation.
"""
struct RichardsImplicitState <: RichardsState
    ψ::Vector{Float}
    θ::Vector{Float}
    ψ_old::Vector{Float}
    θ_old::Vector{Float}
    # specific moisture capacity
    C::Vector{Float}
    # conductivity
    k::Vector{Float}
    k_inter::Vector{Float}
    kΔz⁻¹::Vector{Float}
    Δψ::Vector{Float}  # Δψ/Δz
    # Newton-Raphson work arrays
    dk::Vector{Float}  # dk/dψ
    dS::Vector{Float}  # dS/dψ
    residual::Vector{Float}
    residual_old::Vector{Float}
end

struct Richards1dColumn{P,S}
    parameters::P
    state::S
    solver::NewtonSolver
    tspan::Tuple{Float,Float}
    saveat::Vector{Float}
    saved::Vector{Vector{Float}}
end

"""
    Synchronize the dependent variables (k, C, θ) based on ψ.
"""
function synchronize!(state::RichardsImplicitState, parameters)
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

"""
    Formulate residual and set rhs.

    Use Δt = ∞ for steady-state.
"""
function residual!(state::RichardsImplicitState, Δt)
    @. state.Δψ = @view(state.ψ[2:end]) - @view(state.ψ[1:end-1])
    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])

    # storage
    @. state.residual = state.Δz * (state.θ_old - state.θ) / Δt
    # flow
    @. state.residual += kΔz⁻¹ᵢ₊₁ * Δψᵢ₊₁ - kΔz⁻¹ᵢ₋₁ * Δψᵢ₋₁
    # gravity
    @. state.residual += (@view(state.k_inter[1:end-1]) - @view(state.k_inter[2:end]))
    return
end

"""
    Copy and preserve the old state for time stepping.
"""
function copy_state!(state::RichardsImplicitState)
    state.ψ_old .= state.ψ
    state.θ_old .= state.θ
end

"""
    Restore old state and residual after convergence failure.
"""
function rewind!(state::RichardsImplicitState)
    state.ψ .= state.ψ_old
    state.θ .= state.θ_old
    state.residual .= state.residual_old
end

"""
    Formulate and set coefficients in the matrix.

    Use Δt = ∞ for steady-state.
"""
function jacobian!(linearsolver, state::RichardsImplicitState, parameters, Δt)
    kΔz⁻¹ᵢ₋₁ = @view(state.kΔz⁻¹[1:end-1])
    kΔz⁻¹ᵢ₊₁ = @view(state.kΔz⁻¹[2:end])
    Δz = parameters.Δz

    # Calculate tridiagonal Jacobian matrix coefficients
    dl = linearsolver.M.dl
    d = linearsolver.M.d
    du = linearsolver.M.du
    @. dl = -kΔz⁻¹ᵢ₋₁[2:end]
    @. d = stage.dS / Δt + kΔz⁻¹ᵢ₋₁ + kΔz⁻¹ᵢ₊₁
    @. du = -kΔz⁻¹ᵢ₊₁[1:end-1]

    # Add linearized conductivity terms
    Δψᵢ₋₁ = @view(state.Δψ[1:end-1])
    Δψᵢ₊₁ = @view(state.Δψ[2:end])
    dkᵢ₋₁ = @view(state.dk[1:end-2])
    dkᵢ₊₁ = @view(state.dk[3:end])

    Δψᵢ₋₁l = @view(Δψᵢ₋₁[2:end])
    Δψᵢ₊₁u = @view(Δψᵢ₊₁[1:end-1])
    dkᵢ₋₁l = @view(dkᵢ₋₁[2:end])
    dkᵢ₊₁u = @view(dkᵢ₊₁[1:end-1])

    @. dl += -dkᵢ₋₁l * (Δψᵢ₋₁l / Δz + 1.0)
    @. d += -dkᵢ₋₁ * (Δψᵢ₋₁ / Δz + 1.0) + dkᵢ₊₁ * (Δψᵢ₊₁ / Δz + 1.0)
    @. du += dkᵢ₊₁u * (Δψᵢ₊₁u / Δz + 1.0)

    linearsolver.rhs .= state.residual
    return
end
