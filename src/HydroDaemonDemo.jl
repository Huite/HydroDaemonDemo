module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations

include("utils.jl")
include("forcing.jl")

# Cascade of buckets example
include("reservoirs/equations_reservoirs.jl")
include("reservoirs/explicit_reservoirs.jl")
include("reservoirs/implicit_reservoirs.jl")

end
