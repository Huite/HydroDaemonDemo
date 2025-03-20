# For boundary is nothing

function bottomflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function bottomboundary_jacobian!(J, state, boundary::Nothing)
    return
end

function bottomboundary_residual!(J, state, boundary::Nothing)
    return
end

function topflux(state::RichardsState, parameters::RichardsParameters, boundary::Nothing)
    return 0.0
end

function topboundary_jacobian!(J, state, boundary::Nothing)
    return
end

function topboundary_residual!(J, state, boundary::Nothing)
    return
end

# Precipitation

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    forcing::MeteorologicalForcing,
)
    return -state.forcing[1]
end

function topboundary_residual!(r, state::RichardsState, forcing::MeteorologicalForcing)
    r[end] += state.forcing[1]
end

function topboundary_jacobian!(J, state::RichardsState, forcing::MeteorologicalForcing)
    return
end

# Fixed head, Dirichlet

struct HeadBoundary
    ψ::Float
    k::Float
    dk::Float
end

function HeadBoundary(ψ, constitutive::ConstitutiveRelationships)
    return HeadBoundary(ψ, conductivity(ψ, constitutive), dconductivity(ψ, constitutive))
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
    r,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    r[1] = state.ψ[1] - boundary.ψ
    return
end

function bottomboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    J.d[1] = 1.0
    J.du[1] = 0.0
    return
end

function topflux(
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    Δψ = state.ψ[end] - boundary.ψ
    return boundary.k / (0.5 * parameters.Δz[end]) * Δψ + (state.k[1] - boundary.k)
end

function topboundary_residual!(
    r,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    r[end] = state.ψ[end] - boundary.ψ
end

function topboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::HeadBoundary,
)
    J.d[end] = 1.0
    J.dl[end] = 0.0
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
    r,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    r[1] += state.k[1]
    return
end

function bottomboundary_jacobian!(
    J,
    state::RichardsState,
    parameters::RichardsParameters,
    boundary::FreeDrainage,
)
    J.d[1] += state.dk[1]
    return
end
