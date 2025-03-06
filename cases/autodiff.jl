using ForwardDiff


const Float = Float64


"""Brooks-Corey constitutive relationships"""
struct BrooksCorey
    ψb::Float   # Bubbling pressure (air entry pressure) [L]
    λ::Float    # Pore size distribution index [-]
    ks::Float   # Saturated hydraulic conductivity [L/T]
    θs::Float   # Saturated water content [-]
    θr::Float   # Residual water content [-]
end

function effective_saturation(ψ, bc::BrooksCorey)
    if ψ >= bc.ψb
        return 1.0
    else
        return (bc.ψb / ψ)^(-bc.λ)
    end
end

function conductivity(ψ, bc::BrooksCorey)
    Se = effective_saturation(ψ, bc)
    return bc.ks * Se^(-2 - 3 / bc.λ)
end

function dconductivity(ψ, bc::BrooksCorey)
    wrapper = x -> conductivity(x, bc)
    return ForwardDiff.derivative(wrapper, ψ)
end

bc = BrooksCorey(-0.6, 0.2, 0.1, 0.50, 0.12)
conductivity(-1e2, bc)
dconductivity(-1e2, bc)


using BenchmarkTools


@btime dconductivity(-1e2, bc)
