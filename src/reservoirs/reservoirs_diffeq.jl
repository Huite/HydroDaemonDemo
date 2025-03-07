function reservoir_isoutofdomain(u, p, t)::Bool
    return any(value < 0 for value in u)
end

function reservoir_rhs!(du, u, p, t)
    parameters = p.parameters
    S = u
    q_upstream = 0.0
    for (i, bucket) in enumerate(parameters.buckets)
        q_downstream = flow(bucket, S[i])
        du[i] = (
            precipitation(bucket, p_rate) - evaporation(bucket, S[i], e_rate) + q_upstream - q_downstream
        )
        q_upstream = -q_downstream
    end
end
