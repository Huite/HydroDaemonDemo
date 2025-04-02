struct CascadeState <: State
    S::Vector{Float}
    dS::Vector{Float}
    Sold::Vector{Float}
    forcing::Vector{Float}
end

function primary(state::CascadeState)
    return state.S
end

function righthandside!(du, state::CascadeState, parameters::BucketCascade)
    copyto!(du, state.dS)
    return
end

function prepare_state(p::BucketCascade, initial)
    return CascadeState(copy(initial), zero(initial), copy(initial), zeros(2))
end

function synchronize!(state::CascadeState, parameters)
    return
end

function apply_update!(state::CascadeState, linearsolver, a)
    @. state.S += a * linearsolver.ϕ
    return
end

function copy_state!(state::CascadeState)
    copyto!(state.Sold, state.S)
    return
end

function rewind!(state::CascadeState)
    copyto!(state.S, state.Sold)
    return
end

function force!(state::CascadeState, parameters, t)
    p, e = find_rates(parameters.forcing, t)
    state.forcing[1] = p
    state.forcing[2] = e
    return
end
