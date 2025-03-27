# For boundary is nothing

function bottomflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function bottomboundary_jacobian!(J, state, parameters, boundary::Nothing)
    return
end

function bottomboundary_residual!(J, state, parameters, boundary::Nothing)
    return
end

function topflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function topboundary_jacobian!(J, state, parameters, boundary::Nothing)
    return
end

function topboundary_residual!(J, state, parameters, boundary::Nothing)
    return
end

# Precipitation

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    return state.forcing[1]
end

function topboundary_residual!(
    F,
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    F[end] += state.forcing[1]
end

function topboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    return
end

# Store k value since it never changes
struct HeadBoundary
    ψ::Float
    k::Float
end

function HeadBoundary(ψ, constitutive::ConstitutiveRelationships)
    return HeadBoundary(ψ, conductivity(ψ, constitutive))
end

function bottomflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    Δψ = boundary.ψ - state.ψ[1]
    return boundary.k / (0.5 * parameters.Δz[1]) * Δψ + (boundary.k - state.k[1])
end

function bottomboundary_residual!(
    F,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[1] + boundary.k)
    Δψ = boundary.ψ - state.ψ[1]
    Δz = 0.5 * parameters.Δz[1]
    F[1] += kmean * (Δψ / Δz - 1)
    return
end

function bottomboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[1] + boundary.k)
    Δψ = boundary.ψ - state.ψ[1]
    dk = 0.5 * state.dk[1]
    Δz = 0.5 * parameters.Δz[1]
    J.d[1] += -(kmean / Δz) + dk * (Δψ / Δz - 1)
    return
end

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    Δψ = state.ψ[end] - boundary.ψ
    return boundary.k / (0.5 * parameters.Δz[end]) * Δψ + (state.k[end] - boundary.k)
end

function topboundary_residual!(
    F,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[end] + boundary.k)
    Δψ = boundary.ψ - state.ψ[end]
    #Δz = 0.5 * parameters.Δz[end]
    Δz = parameters.Δz[end]
    F[end] += kmean * (Δψ / Δz + 1)
    return
end

function topboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    kmean = 0.5 * (state.k[end] + boundary.k)
    Δψ = boundary.ψ - state.ψ[end]
    dk = 0.5 * state.dk[end]
    #Δz = 0.5 * parameters.Δz[end]
    Δz = parameters.Δz[end]
    J.d[end] += -(kmean / Δz) + dk * (Δψ / Δz + 1)
    return
end

# Free drainage

struct FreeDrainage end

function bottomflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    return -state.k[1]
end

function bottomboundary_residual!(
    F,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    F[1] -= state.k[1]
    return
end

function bottomboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    J.d[1] -= state.dk[1]
    return
end
