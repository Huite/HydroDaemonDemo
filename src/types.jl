abstract type HydrologicalModel end
abstract type Forcing end
abstract type Parameters end
abstract type State end
abstract type TimeStepper end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

# Model specific types
abstract type Bucket end
abstract type ConstitutiveRelationships end


function prepare_state(parameters, initial)
    error("prepare_state not implemented for $(typeof(parameters))")
end

function primary(state::State)
    error("primary not implemented for $(typeof(state))")
end

function righthandside(state::State)
    error("righthandside not implemented for $(typeof(state))")
end

function jacobian!(J, state::ImplicitState, parameters, Δt)
    error("jacobian! not implemented for $(typeof(state))")
end

function residual!(rhs, state::ImplicitState, parameters, Δt)
    error("residual! not implemented for $(typeof(state))")
end

function copy_state!(state::ImplicitState)
    error("copy_state! not implemented for $(typeof(state))")
end

function rewind!(state::ImplicitState)
    error("rewind! not implemented for $(typeof(state))")
end
