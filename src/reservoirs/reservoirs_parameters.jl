struct Bucket
    area::Float64
    a::Float64
    b::Float64
end

struct BucketCascade <: Parameters
    buckets::Vector{Bucket}
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
end

function BucketCascade(
    area::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    forcing::MeteorologicalForcing,
)
    buckets = [Bucket(_area, _a, _b) for (_area, _a, _b) in zip(area, a, b)]
    return BucketCascade(buckets, forcing, zeros(2))
end
