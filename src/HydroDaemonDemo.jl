module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations

include("state.jl")
include("utils.jl")
include("forcing.jl")

# Cascade of buckets example
include("reservoirs/reservoirs_equations.jl")
include("reservoirs/reservoirs_explicit.jl")
include("reservoirs/reservoirs_implicit.jl")

end
