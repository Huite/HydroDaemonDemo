module HydroDaemonDemo

using Revise
using LinearAlgebra
using DifferentialEquations
# Bencharmking
using BenchmarkTools
# For some basic IO
using CSV
using DataFrames
using Dates
using SparseArrays
using SparseConnectivityTracer: AbstractTracer, TracerSparsityDetector, jacobian_sparsity
import DataInterpolations

# Provide a CustomController
import OrdinaryDiffEqCore: AbstractController
import OrdinaryDiffEqCore:
    stepsize_controller!,
    accept_step_controller,
    step_accept_controller!,
    step_reject_controller!

include("types.jl")

# Newton-Raphson solver
include("solver/linear.jl")
include("solver/relax.jl")
include("solver/timestep.jl")
include("solver/newton.jl")
include("solver/picard.jl")

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
include("fuse/fuse.jl")
include("fuse/fuse070_equations.jl")
include("fuse/fuse550_equations.jl")

# Richards 1D column
include("richards/constitutive/haverkamp.jl")
include("richards/constitutive/mualemvangenuchten.jl")
include("richards/constitutive/spline.jl")
include("richards/richards_parameters.jl")
include("richards/richards_state.jl")
include("richards/richards_equations.jl")

# Case utilities
include("case_helpers/richards_case_helpers.jl")

end
