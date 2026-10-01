# [core]
function precipitation(b::Bucket, rate)
    return b.area * rate
end

# [core]
function flow(b::Bucket, S)
    return b.a * (S / b.area)^b.b
end

# [jacobian]
function dflow(b::Bucket, S)
    if S <= 0
        return 0.0  # No flow change when S is negative
    else
        return b.a * b.b * S^(b.b - 1)
    end
end

# [core]
function waterbalance!(dS, S, parameters::BucketCascade)
    p_rate = parameters.currentforcing[1]
    q_downstream = 0.0
    q_upstream = 0.0
    dS .= 0.0
    dS[1] = precipitation(parameters.buckets[1], p_rate)
    for (i, bucket) in enumerate(parameters.buckets)
        q_downstream = flow(bucket, S[i])

        if parameters.truncate
            q_downstream = min(q_downstream, S[i])
        end

        dS[i] += q_upstream - q_downstream
        q_upstream = q_downstream
    end
    return 0.0, q_downstream
end

# Explicit method

# [explicit]
function explicit_timestep!(state::CascadeState, parameters::BucketCascade, Δt)
    (; dS, S) = state
    q1, q2 = waterbalance!(dS, S, parameters)
    @. state.S += state.dS * Δt
    state.flows[1] += Δt * q1
    state.flows[2] += Δt * q2
    return
end

# Implicit methods

# [implicit]
function residual!(rhs, state::CascadeState, parameters::BucketCascade, Δt)
    (; dS, S, Sold) = state
    waterbalance!(dS, S, parameters)
    # Newton-Raphson needs negative residual
    @. rhs = -(dS - (S - Sold) / Δt)
    return
end

# [jacobian]
function dwaterbalance!(J, S, parameters::BucketCascade)
    dFdSᵢ = get_diagonal(J, parameters.n)
    dFdSᵢ₋₁ = get_lower(J, parameters.n)
    # dFdSᵢ₊₁ = J.du is always zero.
    dq_upstream = 0.0
    for (i, bucket) in enumerate(parameters.buckets)
        # Jacobian terms
        # Lower diagonal J[i, i-1]
        if i > 1
            dFdSᵢ₋₁[i-1] = dq_upstream
        end

        # Diagonal: J[i, i]
        dq = dflow(bucket, S[i])
        dFdSᵢ[i] = -dq
        dq_upstream = dq
    end
    return 0.0, dq_upstream
end

# [jacobian]
function jacobian!(J, state::CascadeState, parameters::BucketCascade, Δt)
    dwaterbalance!(J, state.S, parameters)
    J.d .-= 1.0 / Δt
    return
end

# Wrapped for DifferentialEquations.jl

# [diffeq]
function waterbalance!(du, u, p::DiffEqParams{BucketCascade}, t)
    n = p.parameters.n
    dS = @views du[2:(n+1)]
    S = @views u[2:(n+1)]
    q1, q2 = waterbalance!(dS, S, p.parameters)
    du[1] = q1
    du[end] = q2
    return
end

function dwaterbalance!(J, u, p::DiffEqParams{BucketCascade}, t)
    n = p.parameters.n
    S = @views u[2:(n+1)]
    dq1, dq2 = dwaterbalance!(J, S, p.parameters)
    J.nzval[1] = dq1
    J.nzval[end] = dq2
    return
end

# [diffeq]
function isoutofdomain(u, p::DiffEqParams{BucketCascade}, t)::Bool
    n = p.parameters.n
    S = @views u[2:(n+1)]
    return any(value < 0 for value in S)
end
