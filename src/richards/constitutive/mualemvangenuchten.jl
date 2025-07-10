# Modified Mualem–van Genuchten relations (Ippisch 2006)

struct MualemVanGenuchten <: ConstitutiveRelationships
    a::Float64      # van Genuchten a [1/L]
    n::Float64      # pore‑size distribution parameter
    m::Float64      # usually m = 1 − 1/n
    l::Float64      # pore‑connectivity (Mualem τ), default ~= 0.5
    ks::Float64     # saturated hydraulic conductivity [L/T]
    θs::Float64     # saturated water content
    θr::Float64     # residual water content
    ψe::Float64     # air‑entry suction (> 0)  [L]
    Sc::Float64     # cut‑off saturation factor
    function MualemVanGenuchten(; a, n, m = nothing, l, ks, θs, θr, ψe)
        if isnothing(m)
            m = 1 - 1 / n
        end
        Sc = (1 + abs(a * ψe)^n)^(-m)
        new(a, n, m, l, ks, θs, θr, ψe, Sc)
    end
end

function effective_saturation(ψ, mvg::MualemVanGenuchten)
    if ψ > mvg.ψe
        return 1.0
    end
    return (1 / mvg.Sc) * (1 + (mvg.a * abs(ψ))^mvg.n)^(-mvg.m)
end

function dSe_dψ(ψ, mvg::MualemVanGenuchten)
    if ψ > mvg.ψe
        return 0.0
    end
    absψ = abs(ψ)
    f = 1 + (mvg.a * absψ)^mvg.n
    return (mvg.m * mvg.n * mvg.a^mvg.n / mvg.Sc) * absψ^(mvg.n - 1) * f^(-mvg.m - 1)
end

function moisture_content(ψ, mvg::MualemVanGenuchten)
    return mvg.θr + effective_saturation(ψ, mvg) * (mvg.θs - mvg.θr)
end

"""Specific moisture capacity C = dθ/dψ"""
function specific_moisture_capacity(ψ, mvg::MualemVanGenuchten)
    return dSe_dψ(ψ, mvg) * (mvg.θs - mvg.θr)
end

# Helper function
@inline function _F(x, m)
    return 1 - (1 - x^(1 / m))^m
end

function conductivity(ψ, mvg::MualemVanGenuchten)
    Se = effective_saturation(ψ, mvg)
    if Se >= 1
        Kr = 1.0
    else
        Fc = _F(mvg.Sc, mvg.m)
        Se_ = Se * mvg.Sc
        F = _F(Se_, mvg.m)
        Kr = Se^mvg.l * (F / Fc)^2
    end
    return mvg.ks * Kr
end

"""∂K/∂ψ for Jacobian"""
function dconductivity(ψ, mvg::MualemVanGenuchten)
    if ψ > mvg.ψe
        return 0.0
    end

    Se = effective_saturation(ψ, mvg)
    dSe = dSe_dψ(ψ, mvg)
    Fc = _F(mvg.Sc, mvg.m)
    Se_ = Se * mvg.Sc
    G = 1 - Se_^(1 / mvg.m)
    F = 1 - G^mvg.m
    dF_dSe = mvg.Sc * G^(mvg.m - 1) * Se_^(1 / mvg.m - 1)
    dKr_dSe = mvg.l * Se^(mvg.l - 1) * (F / Fc)^2 + 2 * Se^mvg.l * (F / Fc) * (dF_dSe / Fc)
    return mvg.ks * dKr_dSe * dSe
end
