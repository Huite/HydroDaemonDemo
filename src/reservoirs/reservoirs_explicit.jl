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
        ΔS = max(ΔS, -S[i] / Δt)
        S[i] += ΔS * Δt
        q_upstream = -q_downstream
    end
end

function explicit_run!(cascade::BucketCascade, tstart, tend, Δt; saveat = nothing)
    if isnothing(saveat)
        saveat = Δt
    end
    save_every = Int(round(saveat / Δt))
    n_saves = Int(floor((tend - tstart) / saveat))
    n_timesteps = Int(floor((tend - tstart) / Δt))
    t = tstart

    out = zeros(length(cascade.buckets), n_saves + 1)
    out[:, 1] = cascade.S
    save_index = 2
    for i = 1:n_timesteps
        explicit_timestep!(cascade, t, Δt)
        if i % save_every == 0
            out[:, save_index] = cascade.S
            save_index += 1
        end
        t += Δt
    end
    return out
end
