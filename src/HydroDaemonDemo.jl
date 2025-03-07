module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations

const Float = Float64

# Newton-Raphson solver
include("solver/linear.jl")
include("solver/linesearch.jl")
include("solver/pseudotransient.jl")
include("solver/timestep.jl")
include("solver/newton.jl")

include("state.jl")
include("utils.jl")
include("forcing.jl")
include("model.jl")

# Cascade of buckets example
include("reservoirs/reservoirs_equations.jl")
include("reservoirs/reservoirs_explicit.jl")
include("reservoirs/reservoirs_implicit.jl")

end
