struct ExplicitHydrologicalModel{P<:Parameters,S<:State,T<:TimeStepper} <: HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    tspan::Tuple{Float64,Float64}
    saveat::Vector{Float64}  # frequency
    saved::Matrix{Float64}  # output
    savedflows::Matrix{Float64}  # output
    timestepper::T
end

# [explicit]
function ExplicitHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float64},
    tspan,
    saveat,
    timestepper::TimeStepper,
)
    state = prepare_state(parameters, initial)
    saveat = create_saveat(saveat, parameters.forcing, tspan)
    nstate = length(primary(state))
    nsave = length(saveat)
    saved = zeros(nstate, nsave)
    savedflows = zeros(2, nsave)

    return ExplicitHydrologicalModel(
        parameters,
        state,
        tspan,
        saveat,
        saved,
        savedflows,
        timestepper,
    )
end

# [explicit]
"""
First order explicit (Euler Forward) time integration.
"""
function timestep!(model::ExplicitHydrologicalModel, Δt)
    state = model.state
    parameters = model.parameters
    Δt = compute_timestep_size(model.timestepper, model.state, model.parameters, Δt)
    explicit_timestep!(state, parameters, Δt)
    return Δt
end
