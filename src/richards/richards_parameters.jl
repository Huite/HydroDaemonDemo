struct RichardsParameters{C,T,B} <: Parameters
    constitutive::Vector{C}
    Δz::Vector{Float}
    forcing::MeteorologicalForcing
    bottomboundary::B
    topboundary::T
end
