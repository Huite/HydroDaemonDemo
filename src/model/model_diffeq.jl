"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
struct SavedResults
    saved::Matrix{Float}
    save_idx::Base.RefValue{Int}
end

struct DiffEqParams{P,S}
    parameters::P
    state::S
    results::SavedResults
end

function update_forcing!(integrator)
    (; p, t) = integrator
    force!(p.state, p.parameters, t)
    return
end

@kwdef struct SolverConfig
    alg::Any
    dt::Float
    dtmin::Float
    dtmax::Float
    adaptive::Bool = true
    force_dtmin::Bool = false
    abstol::Float = 1e-6
    reltol::Float = 1e-6
    maxiters::Int = 100
    autodiff::Bool = true
    analytic_jacobian::Bool = false
end

struct DiffEqHydrologicalModel{T}
    integrator::T
    saveat::Vector{Float}
    saved::Matrix{Float}
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

function diffeq_jacobian!(J, u, p::DiffEqParams, t, γ)
    jacobian!(J, p.state, p.parameters, γ)
    return
end

function save_state!(integrator)
    (; u, p) = integrator
    idx = p.results.save_idx[]
    p.results.saved[:, idx] .= u
    p.results.save_idx[] += 1
    return
end

function DiffEqHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float},
    tspan,
    saveat,
    solverconfig::SolverConfig,
)
    (tstart, tend) = tspan
    forcing = parameters.forcing
    saveat = create_saveat(saveat, forcing, tspan)
    pushfirst!(saveat, tstart)

    state = prepare_state(parameters, initial)

    nstate = length(primary(state))
    nsave = length(saveat)
    saved = zeros(nstate, nsave)
    savedresults = SavedResults(saved, Ref(1))
    save_callback = PresetTimeCallback(saveat, save_state!; save_positions = (false, false))

    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))

    callbacks = CallbackSet(forcing_callback, save_callback)
    u0 = primary(state)

    params = DiffEqParams(parameters, state, savedresults)
    problem = ODEProblem(diffeq_rhs!, u0, tspan, params)

    integrator = init(
        problem,
        solverconfig.alg;
        progress = false,
        progress_name = "Simulating",
        progress_steps = 100,
        save_everystep = false,
        callback = callbacks,
        tstops = tstops,
        isoutofdomain = isoutofdomain,
        adaptive = solverconfig.adaptive,
        dt = solverconfig.dt,
        dtmin = solverconfig.dtmin,
        dtmax = something(solverconfig.dtmax, tend),
        force_dtmin = solverconfig.force_dtmin,
        abstol = solverconfig.abstol,
        reltol = solverconfig.reltol,
        maxiters = solverconfig.maxiters,
    )
    return DiffEqHydrologicalModel(integrator, saveat, saved)
end

function run!(model::DiffEqHydrologicalModel)
    DifferentialEquations.solve!(model.integrator)
    return
end

function reset_and_run!(model::DiffEqHydrologicalModel, initial)
    # Wipe results
    model.integrator.p.results.saved .= 0.0
    model.integrator.p.results.save_idx[] = 1
    # Set initial state
    u0 = model.integrator.sol.prob.u0
    u0 .= initial
    reinit!(model.integrator, u0)
    run!(model)
    return
end
