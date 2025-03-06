abstract type State end
abstract type ExplicitState <: State end
abstract type ImplicitState <: State end

# For explicit states
function synchronize!(state::T, parameters) where {T<:ExplicitState}
    error("synchronize! not implemented for $(typeof(state))")
end

# For implicit states
function synchronize!(state::T, parameters) where {T<:ImplicitState}
    error("synchronize! not implemented for $(typeof(state))")
end

function jacobian!(solver, state::T, parameters, Δt) where {T<:ImplicitState}
    error("jacobian! not implemented for $(typeof(state))")
end

function residual!(state::T, Δt) where {T<:ImplicitState}
    error("residual! not implemented for $(typeof(state))")
end

function copy_state!(state::T) where {T<:ImplicitState}
    error("copy_state! not implemented for $(typeof(state))")
end

function rewind!(state::T) where {T<:ImplicitState}
    error("rewind! not implemented for $(typeof(state))")
end
