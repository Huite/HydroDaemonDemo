# [explicit]
# [implicit]
struct CascadeState <: State
    S::Vector{Float64}
    dS::Vector{Float64}
    Sold::Vector{Float64}
    flows::Vector{Float64}
end

# [explicit]
# [implicit]
function primary(state::CascadeState)
    return state.S
end

# [explicit]
# [implicit]
function prepare_state(p::BucketCascade, initial)
    return CascadeState(copy(initial), zero(initial), copy(initial), zeros(2))
end

# [implicit]
function apply_update!(state::CascadeState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

# [implicit]
function copy_state!(state::CascadeState, _)
    copyto!(state.Sold, state.S)
    return
end

# [implicit]
function rewind!(state::CascadeState)
    copyto!(state.S, state.Sold)
    return
end

