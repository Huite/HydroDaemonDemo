abstract type HydrologicalModel end
abstract type Forcing end
abstract type Parameters end
abstract type State end
abstract type TimeStepper end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

# Model specific types
abstract type ConstitutiveRelationships end

function aqueous_saturation(ψ, C::ConstitutiveRelationships)
    return moisture_content(ψ, C) / C.θs
end

function prepare_state(parameters, initial)
    error("prepare_state not implemented for $(typeof(parameters))")
end

function primary(state::State)
    error("primary not implemented for $(typeof(state))")
end

function compute_savedflows!(state::State, parameters::Parameters, Δt)
    return
end

function prepare_ode_function(p::Parameters, nstate, detect_sparsity)
    if detect_sparsity
        J = jacobian_sparsity(
            (du, u) -> waterbalance!(du, u, p),
            zeros(nstate),
            zeros(nstate),
            TracerSparsityDetector(),
        )
    else
        J = Tridiagonal(zeros(nstate - 1), zeros(nstate), zeros(nstate - 1))
    end

    f = ODEFunction(waterbalance!; jac_prototype = J)
    return f
end

function reset!(p::Parameters, u0, initial)
    u0 .= 0.0
    @views u0[1:length(initial)] .= initial
    return
end

function jacobian!(J, state::ImplicitState, parameters, Δt)
    error("jacobian! not implemented for $(typeof(state))")
end

function residual!(rhs, state::ImplicitState, parameters, Δt)
    error("residual! not implemented for $(typeof(state))")
end

function copy_state!(state::ImplicitState, parameters::Parameters)
    error("copy_state! not implemented for $(typeof(state))")
end

function rewind!(state::ImplicitState)
    error("rewind! not implemented for $(typeof(state))")
end
