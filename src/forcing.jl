abstract type Forcing end


struct MeteorologicalForcing
    t::Vector{Float}
    precipitation::Vector{Float}
    evaporation::Vector{Float}
end

function find_rates(forcing::MeteorologicalForcing, t)
    index = searchsortedfirst(forcing.t, t)
    return (forcing.precipitation[index], forcing.evaporation[index])
end
