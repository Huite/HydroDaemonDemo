@kwdef struct MualemVanGenuchten <: ConstitutiveRelationships
    a::Float     # Air entry pressure [1/L]
    n::Float     # Pore size distribution [-]
    m::Float     # Usually set to 1 - 1/n [-]
    l::Float     # Pore connectivity parameter, typically 0.5 [-]
    ks::Float    # Saturated hydraulic conductivity [L/T]
    θs::Float    # Saturated water content [-]
    θr::Float    # Residual water content [-]
end

# Mualem-van Genuchten functions
function effective_saturation(ψ, mvg::MualemVanGenuchten)
    return (1 + (mvg.a * abs(min(ψ, 0.0)))^mvg.n)^(-mvg.m)
end

function moisture_content(ψ, mvg::MualemVanGenuchten)
    Se = effective_saturation(ψ, mvg)
    return mvg.θr + Se * (mvg.θs - mvg.θr)
end

function specific_moisture_capacity(ψ, mvg::MualemVanGenuchten)
    Se = effective_saturation(ψ, mvg)
    dSe_dh =
        (mvg.a^mvg.n) * mvg.m * mvg.n * abs(min(ψ, 0.0))^(mvg.n - 1) * Se^(1 / mvg.m) * Se
    return dSe_dh * (mvg.θs - mvg.θr)
end

function conductivity(ψ, mvg::MualemVanGenuchten)
    Se = effective_saturation(ψ, mvg)
    return mvg.ks * Se^mvg.l * (1 - (1 - Se^(1 / mvg.m))^mvg.m)^2
end

"""dK/dψ for Newton formulation."""
function dconductivity(ψ, mvg::MualemVanGenuchten)
    if ψ >= 0
        return 0.0
    end

    Se = effective_saturation(ψ, mvg)
    dSe_dh = (mvg.a^mvg.n) * mvg.n * mvg.m * abs(ψ)^(mvg.n - 1) * Se^(1 / mvg.m) * Se

    # Term 1: derivative of Se^l
    term1 = mvg.l * Se^(mvg.l - 1) * dSe_dh

    # Term 2: derivative of (1 - (1 - Se^(1/m))^m)^2
    inner = 1 - Se^(1 / mvg.m)
    inner_der = -(1 / mvg.m) * Se^(1 / mvg.m - 1) * dSe_dh
    outer = 1 - inner^mvg.m
    outer_der = -mvg.m * inner^(mvg.m - 1) * inner_der

    return mvg.ks * (term1 * (1 - inner^mvg.m)^2 + 2 * Se^mvg.l * outer * outer_der)
end
