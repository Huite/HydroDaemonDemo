function safepow(base, exponent)
    tol = 1e-12
    return (base < 0 || (exponent < 0 && base < tol)) ? 0.0 : base^exponent
end

function dmin(a, b)
    return a <= b ? 1.0 : 0.0
end

function activation(S, Smax)
    return max(0.0, Float64(S > Smax))
end

function dactivation(S, Smax)
    return 0.0
end

function sigmoid_activation(S, Smax, ω)
    return 1 / (1 + exp(-(S - (Smax - 5*ω))/ω))
end

function dsigmoid_activation(S, Smax, ω)
    a = sigmoid_activation(S, Smax, ω)
    return a*(1 - a)/ω
end

# Note: hypot is numerically more stable than √((a - b)^2 + m)

function min_smooth(a, b, m)
    return 0.5*(a+b - hypot(a-b, m))
end

function dmin_smooth(a, b, m)
    return 0.5*(1 - (a-b)/hypot(a-b, m))
end

function max_smooth(a, b, m)
    return 0.5*(a + b + hypot(a-b, m))
end

function dmax_smooth(a, b, m)
    return 0.5*(1 + (a-b)/hypot(a-b, m))
end

function read_forcing(path)
    df = CSV.read(path, DataFrame)
    df.time = Dates.value.(df.Date - df.Date[1]) / 1.0
    return MeteorologicalForcing(df.time, df.P / 1000.0, df.ET / 1000.0)
end

function okabe_ito_colors()
    # Should be colorblind friendly.
    return Dict(
        :orange => "#E69F00",
        :light_blue => "#56B4E9",
        :green => "#009E73",
        :yellow => "#F0E442",
        :blue => "#0072B2",
        :dark_orange => "#D55E00",
        :pink => "#CC79A7",
        :black => "#000000",
    )
end
