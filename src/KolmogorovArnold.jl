module KolmogorovArnold

using Random
using LinearAlgebra

using NNlib
using LuxCore
using WeightInitializers
using ConcreteStructs

using ChainRulesCore
const CRC = ChainRulesCore

include("utils.jl")
export rbf, rswaf, iqf

# Add KAN
include("kdense.jl")
export KDense

# Lean KAN with base activation
include("kdense_rm.jl")
export KDense_rm

# Lean KAN without base activation
include("kdense_rm_nobase.jl")
export KDense_rm_nobase

# Mult KAN
include("kdense_multKAN.jl")
export KDense_mult

#include("kdense2.jl")
#export KDense2

# include("explicit")
# export GDense

end # module
