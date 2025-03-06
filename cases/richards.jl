
richards_implicit = ImplicitHydrologicalModel(
    richards_params,
    richards_state,
    newton_solver,
    tspan,
    saveat,
    saved,
)
