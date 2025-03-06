abstract type RichardsState end

"""Return the primary state."""
function primary(state::S where {S<:RichardsState})
    return state.ψ
end
