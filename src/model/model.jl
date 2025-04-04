"""
Create saveat vector of times.

    If nothing is provided for saveat, use the forcing times.
    Constrain saveat times such that `tsart < tsave <= tend`.
"""
function create_saveat(saveat, forcing, tspan)::Vector{Float64}
    if isnothing(saveat)
        saveat = forcing.t
    end

    # Remove times beyond the end time as defined by tspan.
    tstart, tend = tspan
    first_index =
        min(searchsortedfirst(saveat, tstart; lt = (a, b) -> a <= b), length(saveat))
    last_index = min(searchsortedfirst(saveat, tend) - 1, length(saveat))
    saveat = saveat[first_index:last_index]
    if isempty(saveat) || saveat[end] != tend
        push!(saveat, tend)
    end

    return saveat
end

function run!(model::HydrologicalModel)
    # Make sure state is up-to-date with initial state and parameters.
    synchronize!(model.state, model.parameters)

    tstart, tend = model.tspan
    tstart, tend = model.tspan
    Δt = model.timestepper.Δt0
    t = tstart

    # Store the initial state.
    model.saved[:, 1] .= primary(model.state)

    tforce = model.parameters.forcing.t[1]
    save_index = 1
    force_index = 1

    while (t < tend) && (!isapprox(t, tend))
        # New forcing
        if isapprox(t, tforce)
            force!(model.state, model.parameters, t)
            force_index += 1
            tforce =
                (force_index <= length(model.parameters.forcing.t)) ?
                model.parameters.forcing.t[force_index] : Inf
        end

        # Limit time step to not overshoot the next critical point
        tsave = model.saveat[save_index]
        t_critical = min(tsave, tforce, tend)
        if (t + Δt) > t_critical
            Δt = t_critical - t
        end
        Δt_actual = timestep!(model, Δt)
        t += Δt_actual

        # Store output
        if isapprox(t, tsave)
            model.saved[:, save_index+1] .= primary(model.state)
            save_index += 1
        end
    end
end

function reset_and_run!(model::HydrologicalModel, initial)
    # Wipe results
    model.saved .= 0.0
    # Set initial state
    primary_state = primary(model.state)
    primary_state .= initial
    run!(model)
    return
end
