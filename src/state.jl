abstract type State end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

function primary(state::State)
    error("primary not implemented for $(typeof(state))")
end

# For explicit states
function synchronize!(state::ExplicitState, parameters)
    error("synchronize! not implemented for $(typeof(state))")
end

# For implicit states
function synchronize!(state::ImplicitState, parameters)
    error("synchronize! not implemented for $(typeof(state))")
end

function jacobian!(solver, state::ImplicitState, parameters, Δt) 
    error("jacobian! not implemented for $(typeof(state))")
end

function residual!(solver, state::ImplicitState, parameters, Δt)
    error("residual! not implemented for $(typeof(state))")
end

function copy_state!(state::ImplicitState)
    error("copy_state! not implemented for $(typeof(state))")
end

function rewind!(state::ImplicitState)
    error("rewind! not implemented for $(typeof(state))")
end
