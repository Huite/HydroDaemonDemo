struct Fuse070State <: State
    S::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function primary(state::Fuse070State)
    return state.S
end

function prepare_state(_::Fuse070Parameters, initial)
    return Fuse070State(copy(initial), copy(initial), zeros(2))
end

function force!(state::Fuse070State, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end

