"""
The solver may find negative values of S, these will be rejected, but `dSdt` is
still evaluated. The value of b or c may require a root; this requires a
complex number in case S < 0, which raises an error. Hence we ensure dSdt is
robust to negative values of S.
"""
function dSdt!(dS, S, params, t)
    fuse = params.modelparams
    p = params.forcing[:p]
    PET = params.forcing[:PET]
    S1 = S[1]
    S2 = S[2]

    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    sf = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    qsx = p * (1 - (1 - sf)^fuse.b)
    e1 = PET * min_smooth(S⁺, 1.0, fuse.m)
    q12 = fuse.ku * sf^fuse.c
    qufof = (p - qsx) * sigmoid_activation(S1, fuse.S1max, fuse.ω)
    qb = fuse.v * S2

    dS[1] = p - qsx - e1 - q12 - qufof
    dS[2] = q12 - qb
    return
end

"""
Fetch the new precipitation and evapotranspiration date from the timeseries.
"""
function update!(forcing, timeseries, t)
    index = searchsortedfirst(timeseries.time, t)
    forcing[:p] = timeseries.P[index]
    forcing[:PET] = timeseries.ET[index]
    return
end

function update_integrator!(integrator)
    update!(integrator.p.forcing, integrator.p.timeseries, integrator.t)
end

function indomain(u, p, t)::Bool
    return any(v < 0 for v in u)
end

##

forcing_df = CSV.read("forcing.csv", DataFrame)
forcing_df.time = (forcing_df.Date .- minimum(forcing_df.Date)) ./ Dates.Day(1)

fuse_params = (
    ϕtens = 0.5,
    S1max = 100.0,
    b = 0.2,
    ku = 30.0,
    c = 0.5,
    v = 0.1,
    m = 0.01,
    ω = S1max * 0.05,  # from FUSE paper
)

##

function adhoc_dSdt!(dS, S, params)
    fuse = params.modelparams
    p = params.forcing[:p]
    PET = params.forcing[:PET]
    S1 = S[1]
    S2 = S[2]

    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    saturation = clamp(S1 / fuse.S1max, 0.0, 1.0)
    qsx = p * (1 - (1 - saturation)^fuse.b)
    e1 = PET * min(S⁺, 1.0)
    q12 = fuse.ku * saturation^fuse.c
    qufof = (p - qsx) * activation(S1, fuse.S1max)
    qb = fuse.v * S2
    dS[1] = p - qsx - e1 - q12 - qufof
    dS[2] = q12 - qb
    return
end


function adhoc_run(parameters, S0, tspan, Δt)
    S = copy(S0)
    dS = zeros(2)
    nt = Int(tspan[2] / Δt)
    output = zeros(2, nt)
    t = 0.0
    for i = 1:nt
        update!(parameters.forcing, parameters.timeseries, t)
        adhoc_dSdt!(dS, S, parameters)
        S[1] = max(S[1] + dS[1] * Δt, 0)
        S[2] = max(S[2] + dS[2] * Δt, 0)
        output[:, i] .= S
        t += Δt
    end
    return output
end


S0 = [50.0, 50.0]
tspan = (0, 250.0) #maximum(forcing_df.time))

forcing = ComponentArray(p = 0.0, PET = 0.0)
parameters = DiffEqParams(forcing_df, forcing, fuse_params)
problem = ODEProblem(dSdt!, S0, tspan, parameters)
solution = solve(
    problem,
    Tsit5(),
    callback = PresetTimeCallback(forcing_df.time, update_integrator!),
    tstops = forcing_df.time,
    isoutofdomain = indomain,
)
Sout = adhoc_run(parameters, S0, tspan, 1.0)
plot(solution)
plot!(Sout[1, :])
plot!(Sout[2, :])

##

function clamp_smooth_cubic(x, lo, hi, m)
    d = (hi - lo)
    @assert m <= (0.5 * d) "Smoothing parameter m must be <= half the interval width"

    # Normalize x to [0,1] interval
    x_norm = (x - lo) / d

    # Scaling factor to maintain 1:1 slope in middle region
    a = 1 / (1 - m)

    if x_norm < 0
        y = 0
    elseif x_norm < m
        # Cubic polynomial for lower smoothing region
        # f(0) = 0, f(m) = a*m - (a-1)/2
        # f'(0) = 0, f'(m) = a
        x_scaled = x / m * a
        y = -x_scaled^3 + 2 * x_scaled^2
    elseif x_norm < (1 - m)
        # Linear region with slope a
        y = a * x_norm + 0.5 * (1 - a)
    elseif x_norm < 1
        # Cubic polynomial for upper smoothing region
        # f(1-m) = a*(1-m) + (1-a)/2, f(1) = 1
        # f'(1-m) = a, f'(1) = 0
        t = (x_norm - (1 - m)) / m
        y = (a * m / 2) * (3t^2 - 2t^3)
    else
        y = 1
    end

    return lo + y * d
end
