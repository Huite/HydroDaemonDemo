function isoutofdomain(u, p::DiffEqParams{CascadeState,BucketCascade}, t)::Bool
    return any(value < 0 for value in u)
end

function rhs!(du, u, p::DiffEqParams{CascadeState,BucketCascade}, t)
    ΔS = du
    S = u
    p_rate = p.state.forcing[1]
    e_rate = p.state.forcing[2]

    q_upstream = 0.0
    for (i, bucket) in enumerate(parameters.buckets)
        q_downstream = flow(bucket, S[i])
        ΔS[i] = (
            precipitation(bucket, p_rate) - evaporation(bucket, S[i], e_rate) + q_upstream - q_downstream
        )
        q_upstream = -q_downstream
    end
end
