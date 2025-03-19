abstract type Relaxation end
abstract type LineSearch <: Relaxation end

struct ScalarRelaxation <: Relaxation
    relax::Float
    function ScalarRelaxation(relax::Float)
        0 <= relax < 1 ||
            throw(ArgumentError("Relaxation parameter must be in [0,1): got $relax"))
        new(relax)
    end
end

function newton_step!(
    relaxation::ScalarRelaxation,
    linearsolver,
    state,
    parameters,
    Δt,
)::Bool
    apply_update!(state, linearsolver, 1.0 - relaxation.relax)
    synchronize!(state, parameters)
    residual!(linearsolver, state, parameters, Δt)
    return true
end

struct SimpleLineSearch <: LineSearch
    a0::Float  # initial step size
    b::Float  # backtracking reduction factor
    c::Float  # Sufficient decrease for Armijo condition
    minstep::Float
    maxiter::Int
end

function SimpleLineSearch(; a0 = 1.0, b = 0.5, c = 1e-4, minstep = 1e-10, maxiter = 5)
    return SimpleLineSearch(a0, b, c, minstep, maxiter)
end


function compute_step(ls::SimpleLineSearch, _, α₂, _, _, _)
    return α₂, max(ls.b * α₂, ls.minstep * α₂)
end

struct CubicLineSearch <: LineSearch
    a0::Float  # initial step size
    c::Float  # Sufficient decrease for Armijo condition
    maxiter::Int
    low::Float  # interpolation bounds
    high::Float  # interpolation bounds
end

function CubicLineSearch(; a0 = 1.0, c = 1e-4, maxiter = 5, low = 0.1, high = 0.5)
    return CubicLineSearch(a0, c, maxiter, low, high)
end

function compute_step(ls::CubicLineSearch, α₁, α₂, L2₀, L2₁, L2₂)
    grad₀ = -L2₀  # specifically for the L2-norm

    div = 1.0 / (α₁^2 * α₂^2 * (α₂ - α₁))
    a = (α₁^2 * (L2₂ - L2₀ - grad₀ * α₂) - α₂^2 * (L2₁ - L2₀ - grad₀ * α₁)) * div
    b = (-α₁^3 * (L2₂ - L2₀ - grad₀ * α₂) + α₂^3 * (L2₁ - L2₀ - grad₀ * α₁)) * div

    # Fall back to quadratic on first iteration.
    if abs(a) < eps(Float)
        a_cubic = grad₀ / (2 * b)
    else
        # discriminant
        d = max(b^2 - 3 * a * grad₀, 0.0)
        # quadratic equation root
        a_cubic = (-b + sqrt(d)) / (3 * a)
    end

    # Bound the step size
    a_cubic = clamp(a_cubic, α₂ * ls.low, α₂ * ls.high)
    return α₂, a_cubic
end

function compute_step(ls::CubicLineSearch, α₁, α₂, L2₀, L2₁, L2₂)
    grad₀ = -L2₀  # specifically for the L2-norm
    
    diff₁ = L2₁ - L2₀ - grad₀ * α₁
    diff₂ = L2₂ - L2₀ - grad₀ * α₂
    
    # Calculate cubic coefficients
    div = 1.0 / (α₁^2 * α₂^2 * (α₂ - α₁))
    a = (α₁^2 * diff₂ - α₂^2 * diff₁) * div
    b = (-α₁^3 * diff₂ + α₂^3 * diff₁) * div
    
    if abs(a) < eps(Float)
        # When a ≈ 0, use quadratic model
        a_cubic = -grad₀ / (2 * b)
    else
        d = max(b^2 - 3 * a * grad₀, 0.0)
        # Stability; instead of: a_cubic = (-b + sqrt(d)) / (3 * a)
        a_cubic = -grad₀ / (b + sign(b) * sqrt(d)) 
    end
    
    # Bound the step size
    a_cubic = clamp(a_cubic, α₂ * ls.low, α₂ * ls.high)
    return α₂, a_cubic
end

function newton_step!(ls::LineSearch, linearsolver, state, parameters, Δt)
    # α₀ = 0.0 (implicit)
    α₁ = 0.0
    α₂ = ls.a0
    # Compute the L2 norm of the residual to check for convergence
    L2₀ = norm(linearsolver.rhs)
    L2₁ = L2₀

    L2best = L2₁
    αbest = α₂

    for _ = 1:ls.maxiter
        # Take a step
        apply_update!(state, linearsolver, α₂)
        synchronize!(state, parameters)
        residual!(linearsolver, state, parameters, Δt)
        L2₂ = norm(linearsolver.rhs)

        # Armijo condition for sufficient decrease
        if L2₂ <= ((1 - ls.c * α₂) * L2₀)
            return true
        end
        if L2₂ < L2best
          L2best = L2₂
          αbest = α₂
        end

        # Undo the step by applying a NEGATIVE update
        apply_update!(state, linearsolver, -α₂)

        # Compute new step size
        α₁, α₂ = compute_step(ls, α₁, α₂, L2₀, L2₁, L2₂)
        L2₁ = L2₂
    end
    # Achieved maximum iterations, use αbest.
    apply_update!(state, linearsolver, αbest)
    synchronize!(state, parameters)
    residual!(linearsolver, state, parameters, Δt)
    return false
end
