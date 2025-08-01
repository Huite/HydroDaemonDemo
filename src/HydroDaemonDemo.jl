module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations
# For some basic IO
using CSV
using DataFrames
using Dates
using SparseArrays
using SparseConnectivityTracer: TracerSparsityDetector, jacobian_sparsity
import DataInterpolations

const Float = Float64

include("types.jl")

# Newton-Raphson solver
include("solver/linear.jl")
include("solver/relax.jl")
include("solver/timestep.jl")
include("solver/newton.jl")

include("utils.jl")
include("forcing.jl")
include("model/model_explicit.jl")
include("model/model_implicit.jl")
include("model/model_diffeq.jl")
include("model/model.jl")

# Cascade of bucket reservoirs
include("reservoirs/reservoirs_parameters.jl")
include("reservoirs/reservoirs_state.jl")
include("reservoirs/reservoirs_equations.jl")

# FUSE-070 conceptual model
include("fuse/fuse070_parameters.jl")
include("fuse/fuse070_state.jl")
include("fuse/fuse070_equations.jl")

# Richards 1D column
include("richards/constitutive/haverkamp.jl")
include("richards/constitutive/mualemvangenuchten.jl")
include("richards/constitutive/spline.jl")
include("richards/richards_parameters.jl")
include("richards/richards_state.jl")
include("richards/richards_equations.jl")

end
