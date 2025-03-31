struct Fuse070State <: State
    S::Vector{Float}
    dS::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function primary(state::Fuse070State)
    return state.S
end

function righthandside(state::Fuse070State)
    return state.dS
end

function prepare_state(_::Fuse070Parameters, initial)
    return Fuse070State(copy(initial), zero(initial), copy(initial), zeros(2))
end

function force!(state::Fuse070State, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end

function synchronize!(state::Fuse070State, parameters)
    return
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
