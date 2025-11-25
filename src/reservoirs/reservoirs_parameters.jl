# [core]
struct Bucket
    area::Float64
    a::Float64
    b::Float64
end

# [core]
struct BucketCascade <: Parameters
    buckets::Vector{Bucket}
    forcing::MeteorologicalForcing
    currentforcing::Vector{Float64}
    n::Int
    truncate::Bool
end

function BucketCascade(
    area::Vector{Float64},
    a::Vector{Float64},
    b::Vector{Float64},
    forcing::MeteorologicalForcing,
    truncate = false,
)
    buckets = [Bucket(_area, _a, _b) for (_area, _a, _b) in zip(area, a, b)]
    return BucketCascade(buckets, forcing, zeros(2), length(buckets), truncate)
end


"""
Analytical solution for block (uniform) input over each timestep.

For uniform input during [0, Δt], the weight for routing is computed using
the integrated CDF of the Gamma distribution: Ψ(t) = t·CDF(t) - τ·n·CDF_{Γ(n+1,τ)}(t)

Weight = [Ψ(T+Δt) - 2Ψ(T) + Ψ(T-Δt)] / Δt
where T is the time lag between input and output intervals.
"""
function analytical_solution(parameters::BucketCascade, forcing::MeteorologicalForcing, Δt)
    @assert allequal(b.area for b in parameters.buckets) "All area values must be the same"
    @assert allequal(b.a for b in parameters.buckets) "All a values must be the same"
    @assert all(b.b .== 1.0 for b in parameters.buckets) "All b values must be 1"

    n = parameters.n
    bucket = parameters.buckets[1]
    area = bucket.area
    a = bucket.a
    flows = forcing.precipitation .* area

    τ = area / a
    gamma_dist = Gamma(Float64(n), τ)
    gamma_dist_plus1 = Gamma(Float64(n + 1), τ)

    # Integrated CDF: Ψ(t) = t·CDF(t) - τ·n·CDF_{Γ(n+1,τ)}(t)
    function integrated_cdf(t)
        return (t <= 0 ? 0.0 : t * cdf(gamma_dist, t) - τ * n * cdf(gamma_dist_plus1, t))
    end

    n_flows = length(flows)
    routed = zeros(n_flows)

    for i = 1:n_flows
        for j = 1:i
            T = (i - j) * Δt  # Time lag from input j to output i
            Ψ_plus = integrated_cdf(T + Δt)
            Ψ_mid = integrated_cdf(T)
            Ψ_minus = integrated_cdf(T - Δt)
            weight = (Ψ_plus - 2*Ψ_mid + Ψ_minus) / Δt
            routed[i] += flows[j] * weight
        end
    end

    return routed
end
