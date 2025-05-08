struct MeteorologicalForcing
    t::Vector{Float}
    precipitation::Vector{Float}
    evaporation::Vector{Float}
end

function find_rates(forcing::MeteorologicalForcing, t)
    index = searchsortedfirst(forcing.t, t)
    return (forcing.precipitation[index], forcing.evaporation[index])
end

function force!(parameters, t)
    p, e = find_rates(parameters.forcing, t)
    parameters.currentforcing[1] = p
    parameters.currentforcing[2] = e
    return
end
