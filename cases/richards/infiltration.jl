import HydroDaemonDemo as HDD
using DifferentialEquations
using Sundials

using CSV
using DataFrames
using Dates


function read_forcing(path)
    df = CSV.read(path, DataFrame)
    rename!(df, "Column1" => "Date")
    df.time = Dates.value.(df.Date - df.Date[1]) / 1.0
    P = df[!, "Precipitation (mm/d)"] / 1000.0
    return HDD.MeteorologicalForcing(df.time, P, zero(P))
end

# Note: units are meters and days!
siltloam = HDD.ModifiedMualemVanGenuchten(
    a = 0.423,
    n = 2.06,
    l = 0.5,
    ks = 0.0496,  # m/d
    θr = 0.131,
    θs = 0.396,
    ψe = -1e-3,
    Ss = 1e-6,
)
spline = HDD.SplineConstitutive(siltloam)
forcing = read_forcing("data/infiltration.dat")

infiltration = HDD.RichardsCase(
    soil = siltloam,
    Δz = 0.1,
    Δztotal = 1.5,
    tend = forcing.t[end] + 1.0,
    ψ0 = HDD.InitialConstant(-3.59),
    bottomboundary = HDD.FreeDrainage(),
    topboundary = nothing,
    forcing = forcing,
)
saveat = collect(0.0:1.0:infiltration.tspan[2])

implicit_solver = HDD.NewtonSolver(
    HDD.LinearSolverThomas(infiltration.parameters.n),
    relax = HDD.ScalarRelaxation(0.5),
)
implicit_result = HDD.benchmark!(
    HDD.implicit_model(infiltration, implicit_solver, HDD.AdaptiveTimeStepper(1.0), saveat),
    infiltration,
)


implicit_result2 = HDD.benchmark!(
    HDD.implicit_model(infiltration, implicit_solver, HDD.FixedTimeStepper(1.0), saveat),
    infiltration,
)

model = HDD.implicit_model(infiltration, implicit_solver, HDD.FixedTimeStepper(0.1), saveat)
HDD.run!(model)
@btime HDD.reset_and_run!(model, infiltration.ψ0)

qndf_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)

qndf_dae_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = QNDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)

qbdf_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = QBDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)
qbdf_dae_result = HDD.benchmark!(
    HDD.diffeq_model_dae(
        infiltration,
        HDD.SolverConfig(alg = QBDF(nlsolve = NLNewton(relax = 0.5)), maxiters = 100_000),
        saveat,
    ),
    infiltration,
)

cvode_result = HDD.benchmark!(
    HDD.diffeq_model(
        infiltration,
        HDD.SolverConfig(alg = CVODE_BDF(jac_upper = 1, jac_lower = 1)),
        saveat,
    ),
    infiltration,
)

lsoda = HDD.benchmark!(
    HDD.diffeq_model(infiltration, HDD.SolverConfig(alg = Rosenbrock23()), saveat),
    infiltration,
)


using OrdinaryDiffEqCore: AbstractController
import OrdinaryDiffEqCore: stepsize_controller!

# Custom controller struct
struct NewtonAdaptiveController
    n_increase::Int
    increase::Float64
    n_decrease::Int
    decrease::Float64
    failure::Float64
    dtmin::Float64

    function NewtonAdaptiveController(;
        n_increase = 5,
        increase = 1.25,
        n_decrease = 15,
        decrease = 0.9,
        failure = 0.5,
        dtmin = 1e-6,
    )
        new(n_increase, increase, n_decrease, decrease, failure, dtmin)
    end
end

# Required interface method for step size control
function stepsize_controller!(integrator, controller::NewtonAdaptiveController, alg)
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


custom_controller = NewtonAdaptiveController()
alg = ImplicitEuler(nlsolve = NLNewton(relax = 0.5))
solverconfig =
    HDD.SolverConfig(alg = alg, controller = custom_controller, maxiters = 100_000)
model = HDD.diffeq_model(infiltration, solverconfig, saveat)

HDD.run!(model)
