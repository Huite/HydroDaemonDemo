function safepow(base, exponent)
    # Handle problematic cases that would produce Inf/NaN
    # when base=0 is exponentiated with a negative number.
    tol = 1e-12
    if exponent < 0 && (abs(base) < tol || abs(base - 1) < tol)
        return 0.0
    end
    return base^exponent
end

function dmin(a, b)
    return a <= b ? 1.0 : 0.0
end

function dclamp(x, lo, hi)
    if x <= lo
        return 0.0
    elseif x < hi
        return 1.0
    else
        return 0.0
    end
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

function max_smooth(a, b, m)
    return 0.5*(a + b + hypot(a-b, m))
end

function dmin_smooth(a, b, m)
    return 0.5*(1 - (a-b)/hypot(a-b, m))
end

function dmax_smooth(a, b, m)
    return 0.5*(1 + (a-b)/hypot(a-b, m))
end

function clamp_smooth(x, lo, hi, m)
    return min_smooth(max_smooth(x, lo, m), hi, m)
end

function dclamp_smooth(x, lo, hi, m)
    return dmin_smooth(max_smooth(x, lo, m), hi, m) * dmax_smooth(x, lo, m)
end

function read_forcing(path)
    df = CSV.read(path, DataFrame)
    df.time = Dates.value.(df.Date - df.Date[1]) / 1.0
    return MeteorologicalForcing(df.time, df.P / 1000.0, df.ET / 1000.0)
end
