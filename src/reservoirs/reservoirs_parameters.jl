struct Bucket
    area::Float
    a::Float
    b::Float
end

struct BucketCascade <: Parameters
    buckets::Vector{Bucket}
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float}
end

function BucketCascade(
    area::Vector{Float},
    a::Vector{Float},
    b::Vector{Float},
    forcing::MeteorologicalForcing,
)
    buckets = [Bucket(_area, _a, _b) for (_area, _a, _b) in zip(area, a, b)]
    return BucketCascade(buckets, forcing, zeros(2))
end
