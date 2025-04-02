"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
struct DiffEqParams{P,S}
    parameters::P
    state::S
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

function SolverConfig(
    dt,
    dtmin,
    dtmax;
    alg = Tsit5(),
    adaptive = true,
    force_dtmin = false,
    abstol = 1e-6,
    reltol = 1e-6,
    maxiters = 100,
)
    return SolverConfig(
        alg,
        adaptive,
        dt,
        dtmin,
        dtmax,
        force_dtmin,
        abstol,
        reltol,
        maxiters,
    )
end

struct DiffEqHydrologicalModel
    problem::ODEProblem
    saveat::Vector{Float}
    tstops::Vector{Float}
    callbacks::CallbackSet
    solverconfig::SolverConfig
end

function diffeq_rhs!(du, u, params::DiffEqParams, t)
    # Copy u into current state
    currentstate = primary(params.state)
    copyto!(currentstate, u)
    waterbalance!(params.state, params.parameters)
    # Copy formulated rhs into du
    righthandside!(du, params.state, params.parameters)
    return
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
    pushfirst!(saveat, 0.0)
    state = prepare_state(parameters, initial)
    params = DiffEqParams(parameters, state)
    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))
    callbacks = CallbackSet(forcing_callback)
    u0 = primary(state)
    problem = ODEProblem(diffeq_rhs!, u0, tspan, params)
    return DiffEqHydrologicalModel(problem, saveat, tstops, callbacks, solverconfig)
end

function run!(model::DiffEqHydrologicalModel)
    config = model.solverconfig
    _, tend = model.problem.tspan
    solution = solve(
        model.problem,
        config.alg;
        progress = false,
        progress_name = "Simulating",
        progress_steps = 100,
        save_everystep = false,
        callback = model.callbacks,
        tstops = model.tstops,
        isoutofdomain = isoutofdomain,
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
    return Matrix(solution)
end
