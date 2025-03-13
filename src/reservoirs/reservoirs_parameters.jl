struct BucketCascade{B<:Bucket} <: Parameters
    buckets::Vector{B}
    forcing::MeteorologicalForcing
end

function bucket_cascade_analytic(
    area::Vector{Float},
    a::Vector{Float},
    b::Vector{Float},
    forcing::MeteorologicalForcing,
)
    buckets = [BucketAnalytic(_area, _a, _b) for (_area, _a, _b) in zip(area, a, b)]
    return BucketCascade(buckets, forcing)
end

function bucket_cascade_autodiff(
    area::Vector{Float},
    a::Vector{Float},
    b::Vector{Float},
    forcing::MeteorologicalForcing,
)
    buckets = [BucketAutodiff(_area, _a, _b) for (_area, _a, _b) in zip(area, a, b)]
    return BucketCascade(buckets, forcing)
end
