struct ImplicitHydrologicalModel{
    P<:Parameters,
    S<:State,
    T<:TimeStepper,
    LS<:LinearSolver,
    R<:Relaxation,
} <: HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    solver::NewtonSolver{LS,R}  # Non-linear Newton-Raphson solver
    tspan::Tuple{Float,Float}
    saveat::Vector{Float}  # frequency
    saved::Matrix{Float}  # output
    timestepper::T
end

function Base.show(io::IO, model::ImplicitHydrologicalModel)
    p_name = Base.typename(typeof(model.parameters)).name
    println(io, "ImplicitHydrologicalModel:")
    println(io, "  Parameters: ", p_name)
    println(io, "  Time span: ", model.tspan)
    println(io, "  Save points: ", length(model.saveat), " points")
    if !isempty(model.saveat)
        println(io, "    Range: [", first(model.saveat), ", ", last(model.saveat), "]")
    end
    println(io, "  Output size: ", size(model.saved))
    println(io, "  Solver: ", typeof(model.solver))
    println(io, "    Linear solver: ", typeof(model.solver).parameters[1])
    println(io, "    Relaxation: ", typeof(model.solver).parameters[2])
    print(io, "  Time stepper: ", typeof(model.timestepper))
end

function ImplicitHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float64},
    solver::NewtonSolver,
    tspan,
    saveat,
    timestepper::TimeStepper,
)
    state = prepare_state(parameters, initial)
    saveat = create_saveat(saveat, parameters.forcing, tspan)
    nstate = length(primary(state))
    nsave = length(saveat) + 1
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

"""
First order implicit (Euler Backward) time integration, with optional:

* Adaptive time stepping
* Line searches or backtracking
* Pseudo-transient continuation (PTC) for steady-state
"""
function timestep!(model::ImplicitHydrologicalModel, Δt)
    copy_state!(model.state, model.parameters)
    converged, n_newton_iter = solve!(model.solver, model.state, model.parameters, Δt)

    while !converged
        Δt = compute_timestep_size(model.timestepper, Δt, converged, n_newton_iter)
        rewind!(model.state)
        converged, n_newton_iter = solve!(model.solver, model.state, model.parameters, Δt)
    end

    # After convergence, compute the recommended next step size based on solver performance?
    #    Δt_next = compute_next_time_step(model.timestepper, Δt, converged, n_newton_iter)
    #    return Δt, Δt_next
    return Δt
end
