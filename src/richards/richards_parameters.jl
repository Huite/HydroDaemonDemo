struct RichardsParameters{C,T,B} <: Parameters
    constitutive::Vector{C}
    Δz::Vector{Float}
    Δz⁻¹::Vector{Float}
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
    # Mutable
    ∇q::Vector{Float}  # ∇q/dt
    θ::Vector{Float}
    C::Vector{Float}
    k::Vector{Float}
    dk::Vector{Float}
    # Internodal data, all size n - 1!
    k_inter::Vector{Float}
    Δψ::Vector{Float}  # ψᵢ - ψᵢ₋₁
    currentforcing::Vector{Float}  # P, ET
end

function RichardsParameters(constitutive, Δz, forcing, bottomboundary, topboundary)
    Δz⁻¹ = @views 2.0 ./ (Δz[1:end-1] .+ Δz[2:end])
    n = length(constitutive)
    return RichardsParameters(
        constitutive,
        Δz,
        Δz⁻¹,
        forcing,
        bottomboundary,
        topboundary,
        n,
        zeros(n),  # ∇q
        zeros(n),  # θ
        zeros(n),  # C
        zeros(n),  # k
        zeros(n),  # dk
        zeros(n - 1),  # k_inter
        zeros(n - 1),  # Δψ
        zeros(2),
    )
end
