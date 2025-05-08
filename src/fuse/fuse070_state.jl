struct Fuse070State <: State
    S::Vector{Float}
    dS::Vector{Float}
    Sold::Vector{Float}
end

function primary(state::Fuse070State)
    return state.S
end

function prepare_state(_::Fuse070Parameters, initial)
    return Fuse070State(copy(initial), zero(initial), copy(initial))
end

function apply_update!(state::Fuse070State, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

function copy_state!(state::Fuse070State)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::Fuse070State)
    copyto!(state.S, state.Sold)
    return
end
