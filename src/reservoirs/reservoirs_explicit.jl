function explicit_timestep!(state::CascadeState, parameters::BucketCascade, Δt)
    S = state.S
    p_rate = state.forcing[1]
    e_rate = state.forcing[2]

    q_upstream = 0.0
    for (i, bucket) in enumerate(parameters.buckets)
        q_downstream = flow(bucket, S[i])
        ΔS = (
            precipitation(bucket, p_rate) - evaporation(bucket, S[i], e_rate) + q_upstream - q_downstream
        )
        S[i] = max(S[i] + ΔS * Δt, 0)
        q_upstream = -q_downstream
    end
end
