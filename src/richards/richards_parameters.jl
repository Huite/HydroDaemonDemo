struct RichardsParameters{C,T,B} <: Parameters
    constitutive::Vector{C}
    Δz::Vector{Float}
    Δz⁻¹::Vector{Float}
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
    n::Int
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
    )
end
