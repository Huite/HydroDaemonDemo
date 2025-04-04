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
    activation = max_smooth(0.0, min(1.0, S / tol), m)
    return b.area * rate * activation
end

function dsmooth_evap_cushion(b::BucketAnalytic, S, rate, tol = 1e2)
    if S < 0.0 || S > tol
        return 0.0  # Derivative is zero outside the transition region
    else
        return b.area * rate * (1.0 / tol)
    end
end

function waterbalance!(state::CascadeState, parameters::BucketCascade)
    S = state.S
    dS = state.dS
    p_rate = state.forcing[1]
    e_rate = state.forcing[2]

    q_upstream = 0.0
    for (i, bucket) in enumerate(parameters.buckets)
        q_downstream = smooth_flow(bucket, S[i])
        dS[i] = (
            precipitation(bucket, p_rate) - smooth_evaporation(bucket, S[i], e_rate) +
            q_upstream - q_downstream
        )
        q_upstream = q_downstream
    end
    return
end

function explicit_timestep!(state::CascadeState, parameters::BucketCascade, Δt)
    waterbalance!(state, parameters)
    @. state.S += state.dS * Δt
    @. state.S = max(state.S, 0)
    return
end

function residual!(rhs, state::CascadeState, parameters::BucketCascade, Δt)
    waterbalance!(state, parameters)
    # Newton-Raphson needs negative residual
    @. rhs = -(state.dS - (state.S - state.Sold) / Δt)
    return
end

function jacobian!(J, state::CascadeState, cascade::BucketCascade, Δt)
    S = state.S
    dFdSᵢ = J.d
    dFdSᵢ₋₁ = J.dl
    # dFdSᵢ₊₁ = J.du is always zero.
    e_rate = state.forcing[2]
    dq_upstream = 0.0
    for (i, bucket) in enumerate(cascade.buckets)
        # Jacobian terms
        # Lower diagonal J[i, i-1]
        if i > 1
            dFdSᵢ₋₁[i-1] = dq_upstream
        end

        # Diagonal: J[i, i]
        dq = dsmooth_flow(bucket, S[i])
        dFdSᵢ[i] = -dq - dsmooth_evaporation(bucket, S[i], e_rate) - 1.0 / Δt
        dq_upstream = dq
    end
end

function isoutofdomain(u, p::DiffEqParams{<:BucketCascade,CascadeState}, t)::Bool
    return any(value < 0 for value in u)
end

function righthandside!(du, state::CascadeState, parameters::BucketCascade)
    copyto!(du, state.dS)
    return
end
