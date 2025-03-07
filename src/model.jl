abstract type HydrologicalModel end

struct ExplicitHydrologicalModel{P,S,T} <: HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    tspan::Tuple{Float,Float}
    saveat::Vector{Float}  # frequency
    saved::Matrix{Float}  # output
    timestepper::T
end

function time_step!(model::ExplicitHydrologicalModel, Δt)
    rhs!(state, parameters, Δt)
    primary(state) .+= state.update
    return Δt
end

struct ImplicitHydrologicalModel{P,S,T, LS, BT} <: HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    solver::NewtonSolver{LS, BT}  # Non-linear Newton-Raphson solver
    tspan::Tuple{Float,Float}
    saveat::Vector{Float}  # frequency
    saved::Matrix{Float}  # output
    timestepper::T
end

function ImplicitHydrologicalModel(
    parameters,
    initial::Vector{Float64},
    solver,
    tspan,
    saveat,
    timestepper,
)
    state = prepare_state(parameters, copy(initial), parameters.forcing)

    if isnothing(saveat)
        saveat = copy(parameters.forcing.t)
    end
    
    nstate = length(primary(state)) 
    nsave = length(saveat)
    saved = zeros(nstate, nsave)

    return ImplicitHydrologicalModel(
        parameters,
        state,
        solver,
        tspan,
        saveat,
        saved,
        timestepper,
    )
end

function time_step!(model::ImplicitHydrologicalModel, Δt)
    converged, n_newton_iter = solve!(model.solver, model.state, model.parameters, Δt)

    while !converged
        Δt = compute_time_step(model.timestepper, Δt, converged, n_newton_iter)
        rewind!(model.state)
        converged, n_newton_iter = solve!(model.solver, model.state, model.parameters, Δt)
    end

    # After convergence, compute the recommended next step size based on solver performance
    # (This allows growth for smooth solutions)
    Δt_next = compute_next_time_step(model.timestepper, Δt, converged, n_newton_iter)
    return Δt, Δt_next
end

function run!(model::HydrologicalModel)
    tstart, tend = model.tspan
    Δt = model.timestepper.Δt0
    t = tstart

    # Store the initial state.
    copyto!(model.saved[:, 1], primary(model.state))

    tsave = model.saveat[1]
    tforce = model.parameters.forcing.t[1]
    save_index = 1
    force_index = 1

    while t < tend
        # New forcing
        if isapprox(t, tforce)
            force!(model.state, model.parameters, t)
            force_index += 1
            tforce = model.parameters.forcing.t[force_index]
        end

        # Limit time step to not overshoot the next critical point
        t_critical = min(tsave, tforce, tend)
        if (t + Δt) > t_critical
            Δt = t_critical - t
        end
        Δt_actual, Δt = time_step!(model, Δt)
        t += Δt_actual

        # Store output
        if isapprox(t, tsave)
            copyto!(model.saved[:, save_index+1], primary(model.state))
            save_index += 1
            tsave = model.saveat[save_index]
        end
    end
end

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
    parameters,
    initial,
    tspan,
    saveat,
    solverconfig,
)
    if isnothing(saveat)
        saveat = copy(parameters.forcing.t)
    end

    forcing = parameters.forcing
    state = prepare_state(parameters, copy(initial), forcing)
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
