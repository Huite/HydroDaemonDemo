struct CascadeState <: State
    S::Vector{Float64}
    dS::Vector{Float64}
    Sold::Vector{Float64}
    flows::Vector{Float64}
end

function primary(state::CascadeState)
    return state.S
end

function prepare_state(p::BucketCascade, initial)
    return CascadeState(copy(initial), zero(initial), copy(initial), zeros(2))
end

function apply_update!(state::CascadeState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

function copy_state!(state::CascadeState, _)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::CascadeState)
    copyto!(state.S, state.Sold)
    return
end

