# [output]
"""
Create saveat vector of times.

    If nothing is provided for saveat, use the forcing times.
    Constrain saveat times such that `tstart <= tsave <= tend`.
"""
function create_saveat(saveat, forcing, tspan)::Vector{Float64}
    if isnothing(saveat)
        saveat = forcing.t
    end

    # Remove times beyond the time span as defined by tspan.
    tstart, tend = tspan
    first_index = searchsortedfirst(saveat, tstart)  # Find first element >= tstart
    last_index = searchsortedlast(saveat, tend)      # Find last element <= tend
    saveat = saveat[first_index:last_index]
    if isempty(saveat) || saveat[end] != tend
        push!(saveat, tend)
    end

    return saveat
end

# [explicit]
# [implicit]
function run!(model::HydrologicalModel)
    tstart, tend = model.tspan
    Δt = model.timestepper.Δt0
    t = tstart

    # Store the initial state.
    model.saved[:, 1] .= primary(model.state)
    # Accumulated flows are zero.
    model.savedflows[:, 1] .= 0.0

    tforce = model.parameters.forcing.t[1]
    save_index = 2  # t = 0.0 is already included.
    force_index = 1

    while (t < tend) && (!isapprox(t, tend))
        # New forcing
        if isapprox(t, tforce)
            force!(model.parameters, t)
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
            model.saved[:, save_index] .= primary(model.state)
            model.savedflows[:, save_index] .= model.state.flows
            save_index += 1
        end
    end
end

function reset_and_run!(model::HydrologicalModel, initial)
    # Wipe results
    model.saved .= 0.0
    model.state.flows .= 0.0
    # Set initial state
    primary_state = primary(model.state)
    copyto!(primary_state, initial)
    run!(model)
    return
end
