"""Haverkamp constitutive relationship."""
@kwdef struct Haverkamp <: ConstitutiveRelationships
    a::Float
    β::Float
    y::Float
    A::Float
    ks::Float
    θs::Float
    θr::Float
end

function conductivity(ψ, h::Haverkamp)
    return h.ks * h.A / (h.A + abs(min(ψ, 0.0))^h.y)
end

"""dK/dψ for Newton formulation."""
function dconductivity(ψ, h::Haverkamp)
    return -h.A * h.ks * h.y * min(ψ, 0.0) * abs(min(ψ, 0.0))^(h.y - 2) /
           (h.A + abs(min(ψ, 0.0))^h.y)^2
end

function moisture_content(ψ, h::Haverkamp)
    return h.a * (h.θs - h.θr) / (h.a + abs(ψ)^h.β) + h.θr
end

function specific_moisture_capacity(ψ, h::Haverkamp)
    return h.a * h.β * (h.θs - h.θr) * abs(ψ)^(h.β - 1) / (h.a + abs(ψ)^h.β)^2
end

function dspecific_moisture_capacity(ψ, h::Haverkamp)
    a, β, θs, θr = h.a, h.β, h.θs, h.θr
    A = a * (θs - θr)
    absψ = abs(ψ)
    num1 = (β - 1) * absψ^(β - 2) * (a + absψ^β)^2
    num2 = 2 * β * ψ * absψ^(2β - 3) * (a + absψ^β)
    denom = (a + absψ^β)^4
    return -A * β * (num1 - num2) / denom
end
