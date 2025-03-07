abstract type Bucket end

struct BucketAnalytic <: Bucket
    area::Float
    a::Float
    b::Float
end

struct BucketAutodiff <: Bucket
    area::Float
    a::Float
    b::Float
end

const threshold = 10.0
const m = 0.1

function precipitation(b::Bucket, rate)
    return b.area * rate
end

function evaporation(b::Bucket, S, rate)
    return b.area * rate * (S > 0)
end

function flow(b::Bucket, S)
    Spos = max(S, 0)
    return b.a * Spos^b.b
end

# Jacobian terms

function devaporation(b::BucketAnalytic, S, rate)
    return 0.0
end

function dflow(b::BucketAnalytic, S)
    if S <= 0
        return 0.0  # No flow change when S is negative
    else
        return b.a * b.b * S^(b.b - 1)
    end
end

# Smoothed terms

function smooth_flow(b::Bucket, S)
    Spos = max_smooth(S, 0, m)
    return b.a * Spos^b.b
end

function dsmooth_flow(b::BucketAnalytic, S)
    dSpos = dmax_smooth(S, 0, m)
    Spos = max_smooth(S, 0, m)
    dflow_dSpos = b.a * b.b * Spos^(b.b - 1)
    return dflow_dSpos * dSpos
end

function smooth_evaporation(b::Bucket, S, rate)
    activation = clamp_smooth(S / threshold, 0, 1, m)
    return b.area * rate * activation
end

function dsmooth_evaporation(b::BucketAnalytic, S, rate)
    dactivation = dclamp_smooth(S / threshold, 0, 1, m)
    return b.area * rate * dactivation * (1 / threshold)
end

function smooth_evap_sigmoid(b::Bucket, S, rate; k = 10.0)
    # k controls transition sharpness
    activation = 1.0 / (1.0 + exp(-k * S))
    return b.area * rate * activation
end

function dsmooth_evap_sigmoid(b::BucketAnalytic, S, rate; k = 10.0)
    activation = 1.0 / (1.0 + exp(-k * S))
    dactivation = k * activation * (1.0 - activation)
    return b.area * rate * dactivation
end

function smooth_evap_cushion(b::Bucket, S, rate, tol = 1e2)
    # Only activate evaporation when S is sufficiently above zero
    activation = max(0.0, min(1.0, S / tol))
    return b.area * rate * activation
end

function dsmooth_evap_cushion(b::BucketAnalytic, S, rate, tol = 1e2)
    if S < 0.0 || S > tol
        return 0.0  # Derivative is zero outside the transition region
    else
        return b.area * rate * (1.0 / tol)
    end
end

struct BucketCascade{B<:Bucket}
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

struct CascadeState
    S::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function primary(state::CascadeState)
    return state.S
end

function prepare_state(p::BucketCascade, initial, forcing)
    return CascadeState(initial, copy(initial), zeros(2))
end
