struct Bucket
    area::Float64
    a::Float64
    b::Float64
end

abstract type Forcing end

struct PrecipitationForcing <: Forcing
    t::Vector{Float64}    
    rate::Vector{Float64}
end

struct EvaporationForcing <: Forcing
    t::Vector{Float64}
    rate::Vector{Float64}
end

function find_rate(forcing<:Forcing, t)
    index = searchsortedfirst(forcing.t, t)
    return forcing.rate[index]
end

function precipitation(b::Bucket, rate)
    return b.area * rate
end

function evaporation(b::Bucket, S, rate)
    if S > 0
        return b.area * rate
    else
        return 0
    end
end

function flow(b::Bucket, S)
    return b.a * S ^ b.b
end

function explicit_time_step!(ΔS, S, t, Δt, buckets, p_forcing, e_forcing)
    p_rate = find_rate(p_forcing, t)
    e_rate = find_rate(e_forcing, t)
    
    q_upstream = 0.0
    for (i, bucket) in enumerate(buckets)
        q_downstream = flow(bucket, S[i])
        ΔS[i] = (
            precipitation(bucket, p_rate)
            + evaporation(bucket, S[i], e_rate)
            + q_upstream
            - q_downstream
        )
        q_upstream = -q_downstream
    end
    
    @. S += ΔS * Δt
end

function run_explicit!(buckets, p_forcing, e_forcing, tstart, tend, Δt)
    n_bucket = length(buckets)
    n_timesteps = Int(floor((tend - tstart) / Δt))
    out = zeros(n_bucket, n_timesteps)
    
    S = zeros(n_bucket)
    ΔS = zeros(n_bucket)
    for i in 1:n_timesteps
        explicit_time_step!(ΔS, S, t, Δt, buckets, p_forcing, e_forcing)
        out[:, i] = S
        t += Δt
    end
    return out
end