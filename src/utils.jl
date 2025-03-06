function min_smooth(a, b, m)
    return 0.5 * (a + b - √((a - b)^2 + m))
end

function dmin_smooth(a, b, m)
    return 0.5 * (1 - (a - b) / √((a - b)^2 + m))
end

function max_smooth(a, b, m)
    return 0.5 * (a + b + √((a - b)^2 + m))
end

function dmax_smooth(a, b, m)
    return 0.5 * (1 + (a - b) / √((a - b)^2 + m))
end

function clamp_smooth(x, lo, hi, m)
    d = (hi - lo)
    @assert m <= (0.5 * d) "Smoothing parameter m must be <= half the interval width"
    x_norm = (x - lo) / d
    a = 1 / (1 - m)

    if x_norm < 0
        y = 0
    elseif x_norm < m
        y = (a / (2 * m)) * x_norm^2
    elseif x_norm < (1 - m)
        y = a * x_norm + 0.5 * (1 - a)
    elseif x_norm < 1
        y = 1 - (a / (2 * m)) * (1 - x_norm)^2
    else
        y = 1
    end

    return lo + y * d
end

function dclamp_smooth(x, lo, hi, m)
    d = (hi - lo)
    @assert m <= (0.5 * d) "Smoothing parameter m must be <= half the interval width"
    x_norm = (x - lo) / d
    a = 1 / (1 - m)

    # Derivative with respect to x_norm
    if x_norm < 0
        dy_dx_norm = 0
    elseif x_norm < m
        dy_dx_norm = (a / m) * x_norm
    elseif x_norm < (1 - m)
        dy_dx_norm = a
    elseif x_norm < 1
        dy_dx_norm = (a / m) * (1 - x_norm)
    else
        dy_dx_norm = 0
    end
    dx_norm_dx = 1 / d

    return dy_dx_norm * dx_norm_dx
end


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

function sigmoid_activation(S, Smax, ω)
    return 1 - 1 / (1 + exp((S - Smax) / ω))
end

function activation(S, Smax)
    return max(0.0, Float64(S > Smax))
end
