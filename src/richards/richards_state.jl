"""
This struct holds the mutable members of the Richards 1D simulation.
"""
struct RichardsState <: State
    ψ::Vector{Float}
    θ::Vector{Float}
    ψ_old::Vector{Float}
    θ_old::Vector{Float}
    # specific moisture capacity
    C::Vector{Float}
    # conductivity
    k::Vector{Float}
    # Internodal data, all size n - 1!
    k_inter::Vector{Float}
    kΔz⁻¹::Vector{Float}
    Δψ::Vector{Float}  # Δψ/Δz
    # Newton-Raphson work arrays
    dk::Vector{Float}  # dk/dψ
    dS::Vector{Float}  # dS/dψ
    # Forcing
    forcing::Vector{Float}
end

"""Return the primary state."""
function primary(state::S where {S<:RichardsState})
    return state.ψ
end

function prepare_state(p::RichardsParameters, initial)
    n = length(p.constitutive)
    return RichardsState(
        copy(initial),  # ψ
        zeros(n),  # θ
        zeros(n),  # ψ_old
        zeros(n),  # θ_old
        zeros(n),  # C
        zeros(n),  # k
        zeros(n - 1),  # k_inter
        zeros(n - 1),  # kΔ⁻¹
        zeros(n - 1),  # Δψ
        zeros(n),  # dk
        zeros(n),  # dS
        zeros(2),
    )
end

function force!(state::RichardsState, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end
