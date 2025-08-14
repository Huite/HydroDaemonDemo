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
    function Fuse070Parameters(; ϕtens, S1max, b, ku, c, v, m, forcing)
        ω = S1max * 0.05  # from FUSE paper
        new(ϕtens, S1max, b, ku, c, v, m, ω, forcing, zeros(2))
    end
end
