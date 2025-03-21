struct ExplicitHydrologicalModel{P<:Parameters,S<:State,T<:TimeStepper} <: HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    tspan::Tuple{Float,Float}
    saveat::Vector{Float}  # frequency
    saved::Matrix{Float}  # output
    timestepper::T
end

function ExplicitHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float},
    tspan,
    saveat,
    timestepper::TimeStepper,
)
    state = prepare_state(parameters, initial)
    saveat = create_saveat(saveat, parameters.forcing, tspan)
    nstate = length(primary(state))
    nsave = length(saveat) + 1
    saved = zeros(nstate, nsave)

    return ExplicitHydrologicalModel(parameters, state, tspan, saveat, saved, timestepper)
end

"""
First order explicit (Euler Forward) time integration.
"""
function timestep!(model::ExplicitHydrologicalModel, Δt)
    state = model.state
    parameters = model.parameters
    synchronize!(model.state, model.parameters)
    Δt = compute_timestep_size(model.timestepper, model.state, model.parameters, Δt)
    explicit_timestep!(state, parameters, Δt)
    return Δt
end
