module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations

const Float = Float64

include("types.jl")

# Newton-Raphson solver
include("solver/linear.jl")
include("solver/relax.jl")
include("solver/pseudotransient.jl")
include("solver/timestep.jl")
include("solver/newton.jl")

include("utils.jl")
include("forcing.jl")
include("model/model_explicit.jl")
include("model/model_implicit.jl")
include("model/model_diffeq.jl")
include("model/model.jl")

# Cascade of bucket reservoirs
include("reservoirs/reservoirs_equations.jl")
include("reservoirs/reservoirs_parameters.jl")
include("reservoirs/reservoirs_state.jl")
include("reservoirs/reservoirs_explicit.jl")
include("reservoirs/reservoirs_implicit.jl")
include("reservoirs/reservoirs_diffeq.jl")

end
