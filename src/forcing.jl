abstract type Forcing end

struct PrecipitationForcing <: Forcing
    t::Vector{Float64}
    rate::Vector{Float64}
end

struct EvaporationForcing <: Forcing
    t::Vector{Float64}
    rate::Vector{Float64}
end

function find_rate(forcing::Forcing, t)
    index = searchsortedfirst(forcing.t, t)
    return forcing.rate[index]
end
