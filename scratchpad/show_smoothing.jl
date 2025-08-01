##
using Plots
##

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
    a = 1 / (1 - 2 * m)

    if x_norm <= 0
        y = 0
    elseif x_norm < m
        # Cubic polynomial for lower smoothing region
        # f(0) = 0, f(m) = a*m
        # f'(0) = 0, f'(m) = a
        t = x_norm / m
        y = a * m * t^2 * (3 - 2 * t)
    elseif x_norm <= (1 - m)
        # Linear region with slope a
        y = a * (x_norm - m) + a * m
    elseif x_norm < 1
        # Cubic polynomial for upper smoothing region
        # f(1-m) = 1-a*m, f(1) = 1
        # f'(1-m) = a, f'(1) = 0
        t = (x_norm - (1 - m)) / m
        y = 1 - a * m * (1 - t)^2 * (3 - 2 * (1 - t))
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

##

threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = sigmoid_activation.(x, threshold, 0.01)
y2 = activation.(x, threshold)
plot(x, y)
plot!(x, y2)

##

x = collect(-0.1:0.01:1.0)
y = min_smooth.(x, threshold, 0.01)
y2 = min.(x, threshold)
plot(x, y)
plot!(x, y2)

##

threshold = 0.3
x = collect(-1.0:0.01:1.0)
y = max_smooth.(x, threshold, 0.01)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)

##


threshold = 0.5
x = collect(-10.0:0.01:10.0)
y = max_smooth.(x, threshold, 0.1)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)

##

function bounded_smooth(x, a, b, m)
    # First, smooth max to ensure x ≥ a
    x_lower_bounded = 0.5 * (x + a + √((x - a)^2 + m))

    # Then, smooth min to ensure x ≤ b
    x_bounded = 0.5 * (x_lower_bounded + b - √((x_lower_bounded - b)^2 + m))

    return x_bounded
end

function positive_bounded_smooth(x, upper_bound, m)
    return bounded_smooth(x, 0, upper_bound, m)
end

# Or a simpler version if you just need positivity with no upper bound
function positive_smooth(x, m)
    return 0.5 * (x + √(x^2 + m))
end


threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = positive_smooth.(x, 0.01)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)

threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = bounded_smooth.(x, 0.0, 0.5, 0.01)
y2 = clamp.(x, 0.0, 0.5)
y3 = clamp_smooth.(x, 0.0, 0.5, 0.1)
plot(x, y)
plot!(x, y2)
plot!(x, y3)


##

function positive_smooth(S, m)
    # This will return 0 when S <= 0, and a smooth approximation of S when S > 0
    return 0.5 * (S + √(S^2 + m))
end

threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = positive_smooth.(x, 0.01)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)


##
function clamp_smooth_cubic(x, lo, hi, m)
    d = (hi - lo)
    @assert m <= (0.5 * d) "Smoothing parameter m must be <= half the interval width"

    # Normalize x to [0,1] interval
    x_norm = (x - lo) / d

    if x_norm <= 0
        y = 0
    elseif x_norm < m
        # Lower smooth transition using cubic Hermite spline
        t = x_norm / m
        y = t^2 * (3 - 2 * t) * m
    elseif x_norm <= (1 - m)
        # Linear portion with slope 1
        y = x_norm
    elseif x_norm < 1
        # Upper smooth transition using cubic Hermite spline
        t = (x_norm - (1 - m)) / m
        y = (1 - m) + t^2 * (3 - 2 * t) * m
    else
        y = 1
    end

    return lo + y * d
end

lo = -1.0
hi = 1.0
x = collect(-1.0:0.01:1.0)
y = clamp.(x, lo, hi)
y2 = clamp_smooth.(x, lo, hi, 0.1)
y3 = clamp_smooth_cubic.(x, lo, hi, 0.1)
plot(x, y)
plot!(x, y2)
plot!(x, y3)
println(y3)


function test(x, y, k)
    return log((exp(k * x)) + exp(k * y)) / k
end


threshold = 0.0
x = collect(-10.0:0.01:10.0)
y = test.(x, 0.0, 10.0)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)


function generalized_mean(a, b, n)
    tmpa = a - min(a)
    @show tmpa
    tmpb = b + min(a)
    return (0.5 * (tmpa^n + tmpb^n))^(1 / n) + min(a)
end


threshold = 0.0
x = collect(-10.0:0.01:10.0)
y = generalized_mean.(x, 0.0, 3)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)



function max_smooth(x, y, ε = 1e-1)
    return 0.5 * (x + y + sqrt((x - y)^2 + ε))
end

threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = max_smooth.(x, 0.0)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)

function smooth_max(x, threshold = 0.0, smoothing = 0.01)
    if x <= threshold
        return 0.0
    elseif x < threshold + smoothing
        # Smooth transition between threshold and threshold+smoothing
        # Using a cubic polynomial with zero value and zero derivative at threshold
        t = (x - threshold) / smoothing
        return (x - threshold) * t^2 * (3.0 - 2.0 * t)
    else
        return x - threshold
    end
end

threshold = 0.0
x = collect(-1.0:0.01:1.0)
y = smooth_max.(x, 0.0, 0.2)
y2 = max.(x, threshold)
plot(x, y)
plot!(x, y2)

function max_smooth(x, lo, hi, m)
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
        return x
    end

    return lo + y * d
end

threshold = 0.0
hi = 0.5
m = 0.1 / hi
x = collect(-1.0:0.01:1.0)
y = max_smooth.(x, 0.0, hi, m)
y2 = max.(x, threshold)
y3 = smooth_max.(x, threshold, m)
plot(x, y)
plot!(x, y2)
plot!(x, y3)
