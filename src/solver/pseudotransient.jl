abstract type PTCMethod end
abstract type PTCStepSelection end

struct IdentityPTC <: PTCMethod end

struct StoragePTC <: PTCMethod end

function apply_ptc!(_::IdentityPTC, linearsolver, ptcΔt)
    linearsolver.M.d .+= 1.0 / ptcΔt
    return
end

function apply_ptc!(_::StoragePTC, linearsolver, ptcΔt)
    linearsolver.M.d .+= state.dS / ptcΔt
    return
end

# Note: initialize oldnorm with residual of y0.
struct SwitchedEvolutionRelaxation <: PTCStepSelection
    Δt::Vector{Float}  # one for each state
    Δt0::Float
    Δtmin::Float
    Δtmax::Float
    oldnorm::Vector{Float}  # for mutability
end

# Note: Initialize prevy and preprevy with initial state.
# The third iteration will be meaningful.
struct TemporalTruncationError <: PTCStepSelection
    Δt::Vector{Float}  # one for each state
    τ::Float
    Δt0::Float
    Δtmin::Float
    Δtmax::Float
    Δtprev::Vector{Float}
    preprevy::Vector{Float}
    prevy::Vector{Float}
end

struct PseudoTransientContinuation{S,M}
    boundmin::Float
    boundmax::Float
    stepselection::S
    method::M
end

const OptionalPTC =
    Union{PseudoTransientContinuation{S,M},Nothing} where {S<:PTCStepSelection,M<:PTCMethod}


function firststepsize!(stepselection::PTCStepSelection)
    stepselection.Δt .= stepselection.Δt0
    return
end

function stepsize!(ser::SwitchedEvolutionRelaxation, state, residual)
    newnorm = norm(residual)
    ser.Δt .*= ser.oldnorm / newnorm
    ser.oldnorm = newnorm
    clamp!.(ser.Δt, ser.Δtmin, ser.Δtmax)
    return
end

function stepsize!(tte::TemporalTruncationError, state, residual)
    y = primary(state)
    for i in eachindex(y)
        # Estimate d²y/dt²
        term1 = (y[i] - tte.prevy[i]) / Δt
        term2 = (tte.prevy[i] - tte.preprevy[i]) / tte.Δtprev[1]
        d2y = 2.0 / (tte.Δt[i] + tte.Δtprev[i]) * (term1 - term2)

        if abs(d2y) > 0  # avoid division by zero
            tte.Δtprev[i] = tte.Δt[i]
            tte.Δt[i] = sqrt(2 * tte.τ * (1 + abs(y[i])) / abs(d2y))
            # else keep Δt as is
        end
    end
    tte.preprevy .= tte.prevy
    tte.prevy .= y
    clamp!.(tte.Δt, tte.Δtmin, tte.Δtmax)
    return
end

"""Check whether any value falls outside of plausible bounds, if so: reject and halve the ptc step."""
function check_ptc(ptc::PseudoTransientContinuation, state)
    if any(v -> !(ptc.boundmin <= v <= bound.max), primary(state))
        ptc.stepselection.Δt *= 0.5
        if any(Δt < ptc.Δtmin for Δt in ptc.stepselection.Δt)
            error("PTC failed to find a suitable step size")
        end
        return false
    end
    return true
end
