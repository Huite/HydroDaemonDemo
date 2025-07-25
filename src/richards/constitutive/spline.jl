struct SplineConstitutive{T <: DataInterpolations.CubicHermiteSpline} <: ConstitutiveRelationships
    θ::T
    k::T
    knots::Vector{Float}
end

function conductivity(ψ, hsc::SplineConstitutive)
    return hsc.k(ψ)
end


function dconductivity(ψ, hsc::SplineConstitutive)
    return DataInterpolations.derivative(hsc.k, ψ, 1)
end


function moisture_content(ψ, hsc::SplineConstitutive)
    return hsc.θ(ψ)
end


function specific_moisture_capacity(ψ, hsc::SplineConstitutive)
    return DataInterpolations.derivative(hsc.θ, ψ, 1)
end

function logknots(ψmin, ψe, offset, nknots)
    # Use log-spaced interpolation points.
    # Approach saturation at ψe, but not quite.
    return vcat(
        -exp10.(range(log10(abs(ψmin)), log10(abs(ψe - offset)), length=nknots)),
        ψe .+ [offset, 2*offset]
    )
end

function SplineConstitutive(
    parameters;
    relative_error=1e-3,
    offset=1e-3,
    ψmin=-1e4,
    nknots=50,
    iter=5,
)
    local θspline, kspline, t, maxerror

    if hasproperty(parameters, :ψe)
        ψe = parameters.ψe
    else
        ψe = 0.0
    end

    for _ = 1:iter
        t = logknots(ψmin, ψe, offset, nknots)
        C = specific_moisture_capacity.(t, Ref(parameters))
        θ = moisture_content.(t, Ref(parameters))
        #θspline = DataInterpolations.CubicHermiteSpline(
        #    C, θ, t, extrapolation = DataInterpolations.ExtrapolationType.Constant
        #)
        θspline = DataInterpolations.PCHIPInterpolation(
            θ, t, extrapolation = DataInterpolations.ExtrapolationType.Constant
        )
        dk = dconductivity.(t, Ref(parameters))
        k = conductivity.(t, Ref(parameters))
        #kspline = DataInterpolations.CubicHermiteSpline(
        #    dk, k, t, extrapolation = DataInterpolations.ExtrapolationType.Constant
        #)
        kspline = DataInterpolations.PCHIPInterpolation(
            k, t, extrapolation = DataInterpolations.ExtrapolationType.Constant
        )

        # Test at three times the density.
        ψtest = -exp10.(range(log10(abs(ψmin)), log10(abs(ψe - 1e-2)), length=nknots * 3))
        θref = moisture_content.(ψtest, Ref(parameters))
        kref = conductivity.(ψtest, Ref(parameters))

        θerror = @. (θspline(ψtest) - θref) / θref
        kerror = @. (kspline(ψtest) - kref) / kref

        maxerror = max(
            maximum(abs.(θerror)),
            maximum(abs.(kerror)),
        )

        if maxerror < relative_error
            break
        else
            nknots = ceil(Int, nknots * 1.5)
        end
    end
    return SplineConstitutive(θspline, kspline, t), maxerror
end