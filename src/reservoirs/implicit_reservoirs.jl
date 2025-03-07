using LinearAlgebra

struct Bucket
    area::Float
    a::Float
    b::Float
end

abstract type Forcing end

struct PrecipitationForcing <: Forcing
    t::Vector{Float}    
    rate::Vector{Float}
end

struct EvaporationForcing <: Forcing
    t::Vector{Float}
    rate::Vector{Float}
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

function converged(r, tolerance)
    maxresidual = maximum(abs(residual) for residual in r)
    return maxresidual < tolerance
end

function implicit_time_step!(A, ΔS, r, S, t, Δt, buckets, p_forcing, e_forcing, maxiter, tolerance)
    p_rate = find_rate(p_forcing, t)
    e_rate = find_rate(e_forcing, t)

    for _=1:maxiter
        # Formulate the system of equations
        n_bucket = length(buckets)
        for (i, bucket) in enumerate(buckets)
            # Right-hand-side
            r[i] = (
                precipitation(bucket, p_rate)
                + evaporation(bucket, S[i], e_rate)
                - flow(bucket, S[i])
            )

            if i > 1
                up = buckets[i - 1]
                r[i] += flow(up, S[i - 1])
                # Lower diagonal
                A.dl[i] = up.a * up.b * S[i - 1]^(up.b - 1)
            end
            
            # Diagonal
            A.d[i] = -bucket.a * bucket.b * S[i]^(bucket.b - 1) - 1.0 / Δt
            
            # Upper diagonal
            if i < n_bucket
                A.du[i] = 0
            end
        end
        
        if converged(r, tolerance)
            @. S += ΔS * Δt
            return
        end
        
        ΔS = A \ r
    end
    error("Failed to converge.")
end

function run_implicit!(buckets, p_forcing, e_forcing, tstart, tend, Δt)
    n_bucket = length(buckets)
    n_timesteps = Int(floor((tend - tstart) / Δt))
    maxiter = 100
    tolerance=1e-6
    
    A = Tridiagonal(
        zeros(n_bucket -1),  # lower diagonal
        zeros(n_bucket),  # diagonal
        zeros(n_bucket -1),  # upper diagonal
    )
    r = zeros(n_bucket)
    
    S = zeros(n_bucket)
    ΔS = zeros(n_bucket)
    out = zeros(n_bucket, n_timesteps)
    for i in 1:n_timesteps
        implicit_time_step!(A, ΔS, r, S, t, Δt, buckets, p_forcing, e_forcing, maxiter, tolerance)
        out[:, i] = S
        t += Δt
    end
    return out
end
