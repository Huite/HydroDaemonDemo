struct ImplicitHydrologicalModel{P<:Parameters,S<:State,T<:TimeStepper,NLS} <:
       HydrologicalModel
    parameters::P  # Physical parameters
    state::S  # State and dependent variables
    solver::NLS  # non-linear solver
    tspan::Tuple{Float64,Float64}
    saveat::Vector{Float64}  # frequency
    saved::Matrix{Float64}  # output
    savedflows::Matrix{Float64}  # output
    timestepper::T
end

function Base.show(io::IO, model::ImplicitHydrologicalModel)
    p_name = Base.typename(typeof(model.parameters)).name
    println(io, "ImplicitHydrologicalModel:")
    println(io, "  Parameters: ", p_name)
    println(io, "  Time span: ", model.tspan)
    println(io, "  Solver: ", typeof(model.solver))
    println(io, "    Linear solver: ", typeof(model.solver).parameters[1])
    if typeof(model.solver) isa NewtonSolver
        println(io, "    Relaxation: ", typeof(model.solver).parameters[2])
    end
    println(io, "  Time stepper: ", typeof(model.timestepper))
    println(io, "  Save points: ", length(model.saveat), " points")
    if !isempty(model.saveat)
        println(io, "    Range: [", first(model.saveat), ", ", last(model.saveat), "]")
    end
    print(io, "  Output size: ", size(model.saved))
end

function ImplicitHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float64},
    solver,
    tspan,
    saveat,
    timestepper::TimeStepper,
)
    state = prepare_state(parameters, initial)
    saveat = create_saveat(saveat, parameters.forcing, tspan)
    nstate = length(primary(state))
    nsave = length(saveat) + 1
    saved = zeros(nstate, nsave)
    savedflows = zeros(2, nsave)
    return ImplicitHydrologicalModel(
        parameters,
        state,
        solver,
        tspan,
        saveat,
        saved,
        savedflows,
        timestepper,
    )
end

"""
First order implicit (Euler Backward) time integration, with optional:

* Adaptive time stepping
* Line searches or backtracking
"""
function timestep!(model::ImplicitHydrologicalModel, Δt)
    copy_state!(model.state, model.parameters)
    converged, n_iter = nonlinearsolve!(model.solver, model.state, model.parameters, Δt)

    while !converged
        Δt = compute_timestep_size(model.timestepper, Δt, converged, n_iter)
        rewind!(model.state)
        converged, n_iter = nonlinearsolve!(model.solver, model.state, model.parameters, Δt)
    end

    # Compute the flows based on the current solution
    compute_savedflows!(model.state, model.parameters, Δt)

    # After convergence, compute the recommended next step size based on solver performance?
    #    Δt_next = compute_next_time_step(model.timestepper, Δt, converged, n_newton_iter)
    #    return Δt, Δt_next
    return Δt
end

function nonlinearsolve!(nonlinearsolver, state, parameters, Δt)
    for i = 1:nonlinearsolver.maxiter
        residual!(nonlinearsolver.linearsolver.rhs, state, parameters, Δt)
        # Check the residual immediately for convergence.
        if converged(nonlinearsolver, primary(state))
            return true, i
        end
        setmatrix!(nonlinearsolver, state, parameters, Δt)
        linearsolve!(nonlinearsolver.linearsolver)
        relaxed_update!(
            nonlinearsolver.relax,
            nonlinearsolver.linearsolver,
            state,
            parameters,
            Δt,
        )
    end
    return false, nonlinearsolver.maxiter
end
