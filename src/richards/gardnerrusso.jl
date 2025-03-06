"""Gardner-Russo constitutive relationships"""

abstract type GardnerRusso <: ConstitutiveRelationships end

struct GardnerRussoAnalytic <: GardnerRusso
    α::Float    # Sorptive number [1/L]
    ks::Float   # Saturated hydraulic conductivity [L/T]
    θs::Float   # Saturated water content [-]
    θr::Float   # Residual water content [-]
end

struct GardnerRussoAutodiff
    α::Float    # Sorptive number [1/L]
    ks::Float   # Saturated hydraulic conductivity [L/T]
    θs::Float   # Saturated water content [-]
    θr::Float   # Residual water content [-]
end

function conductivity(ψ, gr::GardnerRusso)
    return gr.ks * exp(gr.α * min(ψ, 0.0))
end

function moisture_content(ψ, gr::GardnerRusso)
    if ψ >= 0
        return gr.θs
    end
    return gr.θr + (gr.θs - gr.θr) * exp(gr.α * ψ)
end

# Analytic derivatives

function dconductivity(ψ, gr::GardnerRussoAnalytic)
    if ψ >= 0
        return 0.0
    end
    return gr.ks * gr.α * exp(gr.α * ψ)
end

function specific_moisture_capacity(ψ, gr::GardnerRussoAnalytic)
    if ψ >= 0
        return 0.0
    end
    return gr.α * (gr.θs - gr.θr) * exp(gr.α * ψ)
end

# Autodiff derivatives

function dconductivity(ψ, gr::GardnerRussoAutodiff)
    wrapper = x -> conductivity(x, gr)
    return ForwardDiff.derivative(wrapper, ψ)
end

function specific_moisture_capacity(ψ, gr::GardnerRussoAutodiff)
    wrapper = x -> moisture_content(x, gr)
    return ForwardDiff.derivative(wrapper, ψ)
end
