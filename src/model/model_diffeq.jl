# [output]
"""
Wrapper around the (mutable) state and the (immutable) parameters,
as the DifferentialEquations uses a single parameters argument.
"""
mutable struct SavedResults
    saved::Matrix{Float64}
    savedflows::Matrix{Float64}
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

# [diffeq]
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

# [forcing]
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

# [diffeq]
struct DiffEqHydrologicalModel{T}
    integrator::T
    saveat::Vector{Float64}
    saved::Matrix{Float64}
    savedflows::Matrix{Float64}
end

# [diffeq]
function get_parameters(model::DiffEqHydrologicalModel)
    return model.integrator.p.parameters
end

# [diffeq]
function save_state!(integrator)
    (; u, p) = integrator
    n = p.parameters.n
    nflows = 2
    idx = p.results.save_idx
    primary_state = @view u[1:n]
    flows = @view u[(end-nflows+1):end]
    p.results.saved[1:n, idx] .= primary_state
    p.results.savedflows[1:nflows, idx] .= flows
    p.results.save_idx += 1
    return
end

function create_tolvectors(nunknown, nflow, abstol::Vector, reltol::Vector)
    # Do nothing in case tolerances are already specified as vectors.
    return abstol, reltol
end

function create_tolvectors(nunknown, nflow, abstol::Float64, reltol::Float64)
    vector_abstol = fill(abstol, nunknown)
    vector_reltol = fill(reltol, nunknown)
    # Set flow tolerances to a huge number.
    @views vector_abstol[(end-nflow+1):end] .= 1e12
    @views vector_reltol[(end-nflow+1):end] .= 1e12
    return vector_abstol, vector_reltol
end

# [diffeq]
function prepare_jacobian_sparsity(parameters, nunknown)
    return jacobian_sparsity(
        (du, u) -> waterbalance!(du, u, t, parameters),
        zeros(nunknown),
        zeros(nunknown),
        0.0,
        TracerSparsityDetector(),
    )
end

# [diffeq]
function prepare_problem(
    parameters::Parameters,
    savedresults,
    nstate,
    nflow,
    solverconfig,
    initial,
    tspan,
)
    nunknown = nstate + nflow
    if solverconfig.detect_sparsity
        Jpattern = prepare_jacobian_sparsity(parameters, nunknown)
        J = Float64.(Jpattern)
    else
        J = Tridiagonal(zeros(nunknown - 1), zeros(nunknown), zeros(nunknown - 1))
    end

    f = ODEFunction{true}(waterbalance!; jac_prototype = J)
    u0 = zeros(nunknown)
    @views u0[1:length(initial)] .= initial
    params = DiffEqParams(parameters, savedresults)
    problem = ODEProblem(f, u0, tspan, params)

    abstol, reltol =
        create_tolvectors(nunknown, nflow, solverconfig.abstol, solverconfig.reltol)
    return problem, abstol, reltol
end

# [diffeq]
function DiffEqHydrologicalModel(
    parameters::Parameters,
    initial::Vector{Float64},
    tspan,
    saveat,
    solverconfig::SolverConfig,
)
    forcing = parameters.forcing
    saveat = create_saveat(saveat, forcing, tspan)

    nflow = 2
    nstate = length(initial)
    nsave = length(saveat)
    saved = zeros(nstate, nsave)
    savedflows = zeros(nflow, nsave)
    savedresults = SavedResults(saved, savedflows, 1)
    save_callback = PresetTimeCallback(saveat, save_state!; save_positions = (false, false))

    tstops = unique(sort(vcat(forcing.t, saveat)))
    forcing_callback =
        PresetTimeCallback(forcing.t, update_forcing!; save_positions = (false, false))

    callbacks = CallbackSet(forcing_callback, save_callback)

    problem, abstol, reltol = prepare_problem(
        parameters,
        savedresults,
        nstate,
        nflow,
        solverconfig,
        initial,
        tspan,
    )

    # CVODE_BDF requires scalar tolerances.
    if solverconfig.alg isa CVODE_BDF
        abstol = abstol[1]
        reltol = reltol[1]
    end

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
        abstol = abstol,
        reltol = reltol,
        maxiters = solverconfig.maxiters,
    )
    return DiffEqHydrologicalModel(integrator, saveat, saved, savedflows)
end

# [diffeq]
function run!(model::DiffEqHydrologicalModel)
    DifferentialEquations.solve!(model.integrator)
    return
end

function reset_and_run!(model::DiffEqHydrologicalModel, initial)
    # Wipe results
    model.savedflows .= 0.0
    model.saved .= 0.0
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
