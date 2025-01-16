function min_smooth(x, b, m)
  return 0.5 * (x + b - √((x - b)^2 + m))
end

function min_smooth_derivative(x, b, m)
  return -0.25 / √((x - b)^2 + m)
end

function clamp_smooth(x, lo, hi, m)
  d = (hi - lo)
  @assert m <= (0.5 * d)
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

function clamp_smooth_derivative(x, lo, hi, m)
    d = hi - lo
    @assert m <= (0.5 * d)
    a = 1 / (1 - m)
    x_norm = (x - lo) / d
    
    if x_norm < 0
        return 0
    elseif x_norm < m
        return a * (x - lo) / (m*d)
    elseif x_norm < (1 - m)
        return a
    elseif x_norm < 1
        return a * (1 - (x - lo) / d) /m
    else
        return 0
    end
end

function sigmoid_activation(S, Smax, ω)
  return 1 - 1 / (1 + exp((S - Smax) / ω))
end

function sigmoid_activation_derivative(S, Smax, ω)
  a = exp((S - Smax) / ω)
  return exp(a) / (ω * (exp(a) + 1)^2)
end

function dSdt!(dS, S, params, t)
  fuse = params.modelparams
  p = params.forcing[:p]
  PET = params.forcing[:PET]
  S1 = S[1]
  S2 = S[2]

  S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
  sf = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)

  qsx = p * (1 - (1 - sf) ^ fuse.b)
  e1 = PET * min_smooth(S⁺, 1.0, fuse.m)
  q12 = fuse.ku * sf ^ fuse.c
  qufof = (p - qsx) * sigmoid_activation(S1, fuse.S1max, fuse.ω)
  qb = fuse.v * S2

  dS[1] = p - qsx - e1 - q12 - qufof
  dS[2] = q12 - qb
  return
end

function setjac!(jac, params, S)
    fuse = params.modelparams
    p = params.forcing[:p]
    PET = params.forcing[:PET]
    S1 = S[1]
    S2 = S[2]

    # Compute the terms and their derivatives.
    S⁺ = S1 / (fuse.ϕtens * fuse.S1max)
    dS⁺ = 1.0 / (fuse.ϕtens * fuse.S1max)
    sf = clamp_smooth(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    dsf = clamp_smooth_derivative(S1 / fuse.S1max, 0.0, 1.0, fuse.m)
    act = sigmoid_activation(S1, fuse.S1max, fuse.ω)
    dact = sigmoid_activation_derivative(S1, fuse.S1max, fuse.ω)

    # Apply chain rule and product rule as needed.
    de1 = PET * min_smooth_derivative(S⁺, 1.0, fuse.m) * dS⁺
    dq12 = fuse.c * fuse.ku * sf ^ (fuse.c - 1) * dsf
    qsx = p * (1 - (1 - sf) ^ fuse.b)
    dqsx = fuse.p * fuse.b * (1 - sf)^(fuse.b - 1) * dsf * 1.0 / fuse.Smax
    dqufof = -dqsx * act + (fuse.p - qsx) * dact
    dqb = fuse.v

    jac[1, 1] = -de1 - dq12 - dqsx - dqufof
    jac[1, 2] = 0.0
    jac[2, 1] = dq12
    jac[2, 2] = dqb
    return
end
