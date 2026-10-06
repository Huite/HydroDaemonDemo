abstract type HydrologicalModel end
abstract type Forcing end
abstract type Parameters end
abstract type State end
abstract type TimeStepper end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

# Model specific types
abstract type ConstitutiveRelationships end
abstract type Fuse070 <: Parameters end

function get_parameters(model::HydrologicalModel)
    return model.parameters
end

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
    # Compute flows based on the current solution.
    q1, q2 = waterbalance!(state.dS, state.S, parameters)
    state.flows[1] += Δt * q1
    state.flows[2] += Δt * q2
    return
end

function reset!(p::Parameters, u0, initial)
    u0 .= 0.0
    n = length(initial)
    @views u0[2:(n+1)] .= initial
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

get_diagonal(J::Tridiagonal, n) = J.d
get_lower(J::Tridiagonal, n) = J.dl
get_upper(J::Tridiagonal, n) = J.du

# This assumes a flow at the first spot (and at the last)
get_diagonal(J::SparseMatrixCSC, n) = @view J.nzval[2:3:(3n-1)]
get_lower(J::SparseMatrixCSC, n) = @view J.nzval[3:3:(3n-3)]
get_upper(J::SparseMatrixCSC, n) = @view J.nzval[4:3:(3n-2)]
