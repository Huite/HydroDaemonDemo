struct Fuse070Parameters <: Parameters
    ϕtens::Float64
    S1max::Float64
    b::Float64
    ku::Float64
    c::Float64
    v::Float64
    m::Float64
    ω::Float64
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
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
    return Fuse070Parameters(ϕtens, S1max, b, ku, c, v, m, ω, forcing, zeros(2))
end
