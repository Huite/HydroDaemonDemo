"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
mutable struct SavedResults
    saved::Matrix{Float64}
    save_idx::Int
end

function Base.show(io::IO, sr::SavedResults)
    n_vars, n_timesteps = size(sr.saved)
    nsave = sr.save_idx - 1
    print(
        io,
        "SavedResults($(n_vars) variables x $(n_timesteps) time steps, saved: $(nsave)/$(n_timesteps))",
    )
end

struct DiffEqParams{P}
    parameters::P
    results::SavedResults
end

function Base.show(io::IO, dep::DiffEqParams)
    P = typeof(dep.parameters)
    p_name = string(Base.typename(P).name)

    println(io, "DiffEqParams{$p_name}:")
    println(io, "  Parameters: ", dep.parameters)
    print(io, "  Results: ", dep.results)
end

function update_forcing!(integrator)
    (; p, t) = integrator
    force!(p.parameters, t)
    return
end

@kwdef struct SolverConfig
    alg::Any
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-6
    maxiters::Int = 100_000
    detect_sparsity::Bool = false
    controller::Any = nothing
end

struct DiffEqHydrologicalModel{T}
    integrator::T
    saveat::Vector{Float64}
    saved::Matrix{Float64}
end

function save_state!(integrator)
    (; u, p) = integrator
    idx = p.results.save_idx
    p.results.saved[:, idx] .= u
    p.results.save_idx += 1
    return
end

function create_tolvectors(nstate, nflows, abstol, reltol)
    vector_abstol = fill(abstol, nstate)
    vector_reltol = fill(reltol, nstate)
    @views vector_abstol[end-nflows+1:end] .= 1e12
    @views vector_reltol[end-nflows+1:end] .= 1e12
    return vector_abstol, vector_reltol
end


function DiffEqHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float64},
    tspan,
    saveat,
    solverconfig::SolverConfig,
)
    (tstart, tend) = tspan
    forcing = parameters.forcing
    saveat = create_saveat(saveat, forcing, tspan)
    pushfirst!(saveat, tstart)

    nflows = 2
    nstate = length(initial) + nflows

    nsave = length(saveat)
    saved = zeros(nstate, nsave)
    savedresults = SavedResults(saved, 1)
    save_callback = PresetTimeCallback(saveat, save_state!; save_positions = (false, false))

    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))

    callbacks = CallbackSet(forcing_callback, save_callback)
    params = DiffEqParams(parameters, savedresults)

    f = prepare_ode_function(parameters, nstate, solverconfig.detect_sparsity)
    u0 = vcat(initial, zeros(nflows))
    problem = ODEProblem(f, u0, tspan, params)
    # TODO: DAEProblem does not support abstol reltol vectors at the moment.
    #abstol, reltol =
    #    create_tolvectors(nstate, nflows, solverconfig.abstol, solverconfig.reltol)

    integrator = init(
        problem,
        solverconfig.alg;
        controller = solverconfig.controller,
        progress = false,
        progress_name = "Simulating",
        progress_steps = 100,
        save_everystep = false,
        callback = callbacks,
        tstops = tstops,
        isoutofdomain = isoutofdomain,
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
    model.integrator.p.results.save_idx = 1
    # Set initial state
    u0 = model.integrator.sol.prob.u0
    # Dispatch on parameters type for DAE formulation
    reset!(model.integrator.p.parameters, u0, initial)
    # Note: reinit causes allocations.
    # When benchmarking, this will cause some allocations to show up,
    # even if the ODE function is non-allocating.
    reinit!(model.integrator, u0)
    run!(model)
    return
end


function get_kwargs(structure)
    structtype = typeof(structure)
    structname = string(structtype.name.name)
    kwargs =
        [string(f) * "=" * repr(getfield(structure, f)) for f in fieldnames(structtype)]
    return structname, kwargs
end


function Base.show(io::IO, model::DiffEqHydrologicalModel)
    p_name = Base.typename(typeof(model.integrator.p.parameters)).name
    println(io, "DiffEqHydrologicalModel:")
    println(io, "  Parameters: ", p_name)
    tspan = model.integrator.sol.prob.tspan
    println(io, "  Time span: ", tspan)
    println(io, "  Integrator: ", typeof(model.integrator).name.name)
    println(io, "    Algorithm: ", typeof(model.integrator.alg).name.name)
    if hasproperty(model.integrator.alg, :nlsolve)
        name, kwargs = get_kwargs(model.integrator.alg.nlsolve)
        println(io, "    Non-linear solve: ", name, "(", join(kwargs, ", "), ")")
    end
    if hasproperty(model.integrator.alg, :linsolve)
        linsolve_name = typeof(model.integrator.alg.linsolve).name.name
        if linsolve_name == :Nothing
            linsolve_name = "Default"
        end
        println(io, "    Linear solve: ", linsolve_name)
    end

    if hasproperty(model.integrator.opts, :controller)
        name, kwargs = get_kwargs(model.integrator.opts.controller)
        println(io, "    Time step controller: ", name, "(", join(kwargs, ", "), ")")
    end

    println(io, "  Save points: ", length(model.saveat), " points")
    if !isempty(model.saveat)
        println(io, "    Range: [", first(model.saveat), ", ", last(model.saveat), "]")
    end
    # Output information
    print(io, "  Output size: ", size(model.saved))
end

@kwdef struct CustomController <: AbstractController
    n_increase::Int = 5
    increase::Float64 = 1.25
    n_decrease::Int = 15
    decrease::Float64 = 0.9
    failure::Float64 = 0.5
    dtmin::Float64 = 1e-6
end

# Required interface method for step size control
function stepsize_controller!(integrator, controller::CustomController, alg)
    # Get the current algorithm's statistics
    # For implicit methods, we need to access Newton iteration info

    # Check if the step was successful
    if integrator.sol.retcode == :Success || integrator.accept_step
        # Step succeeded - check Newton iterations if available
        n_newton_iter = get_newton_iterations(integrator, alg)

        if n_newton_iter < controller.n_increase
            # Few iterations - increase step size
            new_dt = integrator.dt * controller.increase
        elseif n_newton_iter > controller.n_decrease
            # Many iterations - decrease step size
            new_dt = integrator.dt * controller.decrease
        else
            # Acceptable number of iterations - keep step size
            new_dt = integrator.dt
        end
    else
        # Step failed - reduce step size significantly
        new_dt = integrator.dt * controller.failure
        integrator.force_stepfail = true
    end

    # Enforce minimum step size
    if new_dt < controller.dtmin
        error("Time step below dtmin: $(new_dt) < $(controller.dtmin)")
    end

    # Set the new step size
    integrator.dt = new_dt
    integrator.dtcache = new_dt

    return nothing
end

# These methods are required by the AbstractController interface
function accept_step_controller(integrator, controller::CustomController)
    # Called when a step is accepted - can be used for bookkeeping
    return true
end

function reject_step_controller(integrator, controller::CustomController)
    # Called when a step is rejected - can be used for bookkeeping
    return true
end

# Additional required methods for step control
function step_accept_controller!(integrator, controller::CustomController, alg, q)
    # Called when accepting a step - handle step size update
    # Return the step size scaling factor
    if q === nothing
        # Use our custom Newton-based logic
        n_newton_iter = get_newton_iterations(integrator, alg)

        if n_newton_iter < controller.n_increase
            # Few iterations - increase step size
            return controller.increase
        elseif n_newton_iter > controller.n_decrease
            # Many iterations - decrease step size  
            return controller.decrease
        else
            # Acceptable number of iterations - keep step size
            return 1.0
        end
    else
        # Use the algorithm's computed scaling factor
        return q
    end
end

function step_reject_controller!(integrator, controller::CustomController, alg)
    # Called when rejecting a step - return aggressive reduction factor
    # Note: ODE version takes only 3 arguments (no q parameter)
    return controller.failure
end

# Helper function to extract Newton iteration count
function get_newton_iterations(integrator, alg)
    # This depends on the specific algorithm being used
    # For Rosenbrock methods, check if stats are available
    if hasfield(typeof(integrator), :stats) &&
       hasfield(typeof(integrator.stats), :nnonliniter)
        return integrator.stats.nnonliniter
    elseif hasfield(typeof(integrator.cache), :nl_iters)
        return integrator.cache.nl_iters
    elseif hasfield(typeof(integrator.cache), :newton_iters)
        return integrator.cache.newton_iters
    else
        # Fallback: use a reasonable default or try to estimate
        # from other available metrics
        if hasfield(typeof(integrator), :stats) && hasfield(typeof(integrator.stats), :nf)
            # Rough estimate: each Newton iteration typically requires 1-2 function evaluations
            return max(1, integrator.stats.nf ÷ 2)
        else
            return 5  # Default assumption
        end
    end
end
