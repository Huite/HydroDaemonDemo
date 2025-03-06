"""
All additional information we require,
available through the DifferentialEquations interface.
"""
struct DiffEqCascadeParameters{B<:Bucket}
    cascade::BucketCascade{B}
    currentforcing::Vector{Float}
end

function dSdt!(du, u, p, t)
    p_rate = p.currentforcing[1]
    e_rate = p.currentforcing[2]

    dS = du
    S = u
    q_upstream = 0.0
    for (i, bucket) in enumerate(p.cascade.buckets)
        q_downstream = flow(bucket, S[i])
        dS[i] = (
            precipitation(bucket, p_rate) - evaporation(bucket, S[i], e_rate) + q_upstream - q_downstream
        )
        q_upstream = -q_downstream
    end
    return
end

function update_integrator(integrator)
    p = integrator.p
    t = integrator.t

    p.currentforcing[1] = find_rate(cascade.precipitation, t)
    p.currentforcing[2] = find_rate(cascade.evaporation, t)
    return
end

function outofdomain(u, p, t)::Bool
    return any(value < 0 for value in u)
end

function diffeq_run!(bucketparams::DiffEqCascadeParameters, tstart, tend; saveat = nothing)
    tspan = (start, tend)
    problem = ODEProblem(dSdt!, bucketparams.cascade.S0, tspan, bucketparams)
    tstops = cascade.forcing.precipitation.t
    solution = solve(
        problem,
        Tsit5(),
        callback = PresetTimeCallback(tstops, update_forcing!),
        tstops = tstops,
        isoutofdomain = outofdomain,
        saveat = saveat,
    )
    return Array(solution)
end
