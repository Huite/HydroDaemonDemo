"""Haverkamp constitutive relationship."""
struct Haverkamp <: ConstitutiveRelationships
    α::Float
    β::Float
    γ::Float
    A::Float
    ks::Float
    θs::Float
    θr::Float
end

function Haverkamp(; α, β, γ, A, ks, θs, θr)
    return Haverkamp(α, β, γ, A, ks, θs, θr)
end

function conductivity(ψ, h::Haverkamp)
    return h.ks * h.A / (h.A + abs(min(ψ, 0.0))^h.γ)
end

"""dK/dψ for Newton formulation."""
function dconductivity(ψ, h::Haverkamp)
    return -h.A * h.ks * h.γ * min(ψ, 0.0) * abs(min(ψ, 0.0))^(h.γ - 2) /
           (h.A + abs(min(ψ, 0.0))^h.γ)^2
end

function moisture_content(ψ, h::Haverkamp)
    return h.α * (h.θs - h.θr) / (h.α + abs(ψ)^h.β) + h.θr
end

function specific_moisture_capacity(ψ, h::Haverkamp)
    return h.α * h.β * (h.θs - h.θr) * abs(ψ)^(h.β - 1) / (h.α + abs(ψ)^h.β)^2
end

function dspecific_moisture_capacity(ψ, h::Haverkamp)
    α, β, θs, θr = h.α, h.β, h.θs, h.θr
    A = α * (θs - θr)
    absψ = abs(ψ)
    num1 = (β - 1) * absψ^(β - 2) * (α + absψ^β)^2
    num2 = 2 * β * ψ * absψ^(2β - 3) * (α + absψ^β)
    denom = (α + absψ^β)^4
    return -A * β * (num1 - num2) / denom
end
