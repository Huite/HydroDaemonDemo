function isoutofdomain(u, p, t)::Bool
    return any(value < 0 for value in u)
end

function model_rhs!(du, u, p::DiffEqParams{<:BucketCascade,CascadeState}, t)
    ΔS = du
    S = u
    p_rate = p.state.forcing[1]
    e_rate = p.state.forcing[2]

    q_upstream = 0.0
    for (i, bucket) in enumerate(p.parameters.buckets)
        q_downstream = smooth_flow(bucket, S[i])
        ΔS[i] = (
            precipitation(bucket, p_rate) - smooth_evaporation(bucket, S[i], e_rate) +
            q_upstream - q_downstream
        )
        q_upstream = q_downstream
    end
    return
end
