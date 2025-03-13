struct CascadeState <: State
    S::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function primary(state::CascadeState)
    return state.S
end

function prepare_state(p::BucketCascade, initial, forcing)
    return CascadeState(copy(initial), copy(initial), zeros(2))
end

function force!(state::CascadeState, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end
