module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations
# For some basic IO
using CSV
using DataFrames
using Dates

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

# FUSE-070 conceptual model
include("fuse/fuse070_parameters.jl")
include("fuse/fuse070_state.jl")
include("fuse/fuse070_explicit.jl")
include("fuse/fuse070_implicit.jl")
include("fuse/fuse070_diffeq.jl")

# Richards 1D column
include("richards/constitutive/haverkamp.jl")
include("richards/richards_parameters.jl")
include("richards/richards_state.jl")
include("richards/richards_equations.jl")
include("richards/richards_explicit.jl")

end
