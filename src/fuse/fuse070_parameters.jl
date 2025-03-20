struct Fuse070Parameters <: Parameters
    ϕtens::Float
    S1max::Float
    b::Float
    ku::Float
    c::Float
    v::Float
    m::Float
    ω::Float
    forcing::MeteorologicalForcing
end

function Fuse070Parameters(forcing)
    ϕtens = 0.5
    S1max = 100.0
    b = 0.2
    ku = 30.0
    c = 0.5
    v = 0.1
    m = 0.01
    ω = S1max * 0.05  # from FUSE paper
    return Fuse070Parameters(ϕtens, S1max, b, ku, c, v, m, ω, forcing)
end
