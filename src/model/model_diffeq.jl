"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
struct SavedResults
    saved::Matrix{Float64}
    save_idx::Base.RefValue{Int}
end

function Base.show(io::IO, sr::SavedResults)
    n_vars, n_timesteps = size(sr.saved)
    nsave = sr.save_idx[] - 1
    print(
        io,
        "SavedResults($(n_vars) variables x $(n_timesteps) timesteps, saved: $(nsave)/$(n_timesteps))",
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
    force_dtmin::Bool = false
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-6
    maxiters::Int = 100
    autodiff::Bool = true
    detect_sparsity::Bool = false
end

struct DiffEqHydrologicalModel{T}
    integrator::T
    saveat::Vector{Float64}
    saved::Matrix{Float64}
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
    initial::Vector{Float64},
    tspan,
    saveat,
    solverconfig::SolverConfig,
)
    (tstart, tend) = tspan
    forcing = parameters.forcing
    saveat = create_saveat(saveat, forcing, tspan)
    pushfirst!(saveat, tstart)

    nstate = length(initial)
    nsave = length(saveat)
    saved = zeros(nstate, nsave)
    savedresults = SavedResults(saved, Ref(1))
    save_callback = PresetTimeCallback(saveat, save_state!; save_positions = (false, false))

    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))

    callbacks = CallbackSet(forcing_callback, save_callback)
    params = DiffEqParams(parameters, savedresults)

    f = prepare_ode_function(parameters, nstate, solverconfig.detect_sparsity)
    problem = ODEProblem(f, initial, tspan, params)

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
    # Dispatch on parameters type for DAE formulation
    reset!(model.integrator.p.parameters, u0, initial)
    # Note: reinit causes allocations.
    # When benchmarking, this will cause some allocations to show up,
    # even if the ODE function is non-allocating.
    reinit!(model.integrator, u0)
    run!(model)
    return
end

function Base.show(io::IO, model::DiffEqHydrologicalModel)
    p_name = Base.typename(typeof(model.integrator.p.parameters)).name
    println(io, "DiffEqHydrologicalModel:")
    println(io, "  Parameters: ", p_name)
    tspan = model.integrator.sol.prob.tspan
    println(io, "  Time span: ", tspan)
    println(io, "  Save points: ", length(model.saveat), " points")
    if !isempty(model.saveat)
        println(io, "    Range: [", first(model.saveat), ", ", last(model.saveat), "]")
    end
    # Output information
    println(io, "  Output size: ", size(model.saved))
    println(io, "  Integrator: ", typeof(model.integrator).name.name)
    println(io, "    Algorithm: ", typeof(model.integrator.alg).name.name)
    linsolve_name = typeof(model.integrator.alg.linsolve).name.name
    if linsolve_name == :Nothing
        linsolve_name = "Default"
    end
    println(io, "    Linear solve: ", linsolve_name)
end
