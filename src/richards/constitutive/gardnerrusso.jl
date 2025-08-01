"""Gardner-Russo constitutive relationships"""

struct GardnerRusso <: GardnerRussoRelationships
    a::Float    # Sorptive number [1/L]
    ks::Float   # Saturated hydraulic conductivity [L/T]
    θs::Float   # Saturated water content [-]
    θr::Float   # Residual water content [-]
end

function conductivity(ψ, gr::GardnerRusso)
    return gr.ks * exp(gr.a * min(ψ, 0.0))
end

function moisture_content(ψ, gr::GardnerRusso)
    if ψ >= 0
        return gr.θs
    end
    return gr.θr + (gr.θs - gr.θr) * exp(gr.a * ψ)
end

# Analytic derivatives

function dconductivity(ψ, gr::GardnerRusso)
    if ψ >= 0
        return 0.0
    end
    return gr.ks * gr.a * exp(gr.a * ψ)
end

function specific_moisture_capacity(ψ, gr::GardnerRusso)
    if ψ >= 0
        return 0.0
    end
    return gr.a * (gr.θs - gr.θr) * exp(gr.a * ψ)
end
