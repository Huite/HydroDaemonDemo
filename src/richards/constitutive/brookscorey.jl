abstract type BrooksCorey <: ConstitutiveRelationships end


"""Brooks-Corey constitutive relationships"""
struct BrooksCoreyAnalytic <: BrooksCorey
    ψb::Float64  # Bubbling pressure (air entry pressure) [L]
    λ::Float64   # Pore size distribution index [-]
    ks::Float64  # Saturated hydraulic conductivity [L/T]
    θs::Float64  # Saturated water content [-]
    θr::Float64  # Residual water content [-]
end

struct BrooksCoreyAutodiff <: BrooksCorey
    ψb::Float64  # Bubbling pressure (air entry pressure) [L]
    λ::Float64   # Pore size distribution index [-]
    ks::Float64  # Saturated hydraulic conductivity [L/T]
    θs::Float64  # Saturated water content [-]
    θr::Float64  # Residual water content [-]
end

function effective_saturation(ψ, bc::BrooksCorey)
    if ψ >= bc.ψb
        return 1.0
    end
    return bc.ψb / abs(ψ)^bc
end

function moisture_content(ψ, bc::BrooksCorey)
    Se = effective_saturation(ψ, bc)
    return bc.θr + Se * (bc.θs - bc.θr)
end

function conductivity(ψ, bc::BrooksCorey)
    if ψ >= bc.ψb
        return bc.ks
    end
    return bc.ks * (bc.ψb / abs(ψ))^(2 + 3 * bc.λ)
end

# Analytic derivatives

function dconductivity(ψ, bc::BrooksCoreyAnalytic)
    if ψ >= 0
        return 0.0
    end
    Se = effective_saturation(ψ, bc)
    return bc.ks * (2 + 3 / bc.λ) * Se^(1 + 3 / bc.λ) * bc.λ / abs(ψ)
end

function specific_moisture_capacity(ψ, bc::BrooksCoreyAnalytic)
    if ψ >= 0
        return 0.0
    end
    Se = effective_saturation(ψ, bc)
    return bc.λ * (bc.θs - bc.θr) * Se / abs(ψ)
end

# Autodiff derivatives

function dconductivity(ψ, bc::BrooksCoreyAutodiff)
    wrapper = x -> conductivity(x, bc)
    return ForwardDiff.derivative(wrapper, ψ)
end

function specific_moisture_capacity(ψ, bc::BrooksCoreyAutodiff)
    wrapper = x -> moisture_content(x, bc)
    return ForwardDiff.derivative(wrapper, ψ)
end
