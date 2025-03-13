"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
struct DiffEqParams{S,P}
    state::S
    parameters::P
end

function update_forcing!(integrator)
    (; p, t) = integrator
    force!(p.state, p.parameters, t)
    return
end

struct SolverConfig
    alg::Any
    adaptive::Bool
    dt::Float
    dtmin::Float
    dtmax::Float
    force_dtmin::Bool
    abstol::Float
    reltol::Float
    maxiters::Int
end

struct DiffEqHydrologicalModel
    problem::ODEProblem
    saveat::Vector{Float}
    tstops::Vector{Float}
    callbacks::CallbackSet
    solverconfig::SolverConfig
    # isoutofdomain::Function  # use a closure instead, dispatch on parameters or state?
end

function DiffEqHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float},
    tspan,
    saveat,
    solverconfig::SolverConfig,
)
    forcing = parameters.forcing
    saveat = create_saveat(saveat, forcing, tspan)
    state = prepare_state(parameters, initial, forcing)
    params = DiffEqParams(buckets, state)
    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))
    callbacks = CallbackSet(forcing_callback)
    problem = ODEProblem(rhs!, state.S, tspan, params)
    return HydrologicalModel(problem, saveat, tstops, callbacks, solverconfig)
end

function run!(model::DiffEqHydrologicalModel)
    config = model.solverconfig
    _, tend = model.problem.tspan

    model.solution = solve(
        model.problem,
        config.alg;
        progress = false,
        progress_name = "Simulating",
        progress_steps = 100,
        save_everystep = false,
        callback = model.callbacks,
        tstops = model.tstops,
        isoutofdomain = model.isoutofdomain,
        adaptive = config.adaptive,
        dt = config.dt,
        dtmin = config.dtmin,
        dtmax = something(config.dtmax, tend),
        force_dtmin = config.force_dtmin,
        abstol = config.abstol,
        reltol = config.reltol,
        maxiters = config.maxiters,
        saveat = model.saveat,
    )
    return
end
