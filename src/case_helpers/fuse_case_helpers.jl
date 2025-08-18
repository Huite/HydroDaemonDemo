const Bounds = NamedTuple{(:lower, :upper),Tuple{Float64,Float64}}

"""
This implements only the bounds required by FUSE070 and FUSE550 models.
"""
struct FuseParameterBounds
    S1max::Bounds
    S2max::Bounds
    ϕtens::Bounds
    r1::Bounds
    ku::Bounds
    c::Bounds
    ki::Bounds
    ks::Bounds
    n::Bounds
    v::Bounds
    Acmax::Bounds
    b::Bounds
    μt::Bounds
end
