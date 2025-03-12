struct ImplicitHydrologicalModel{P<:Parameters,S<:State,T<:TimeStepper, LS<:LinearSolver, BT<:LineSearch} <: HydrologicalModel
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