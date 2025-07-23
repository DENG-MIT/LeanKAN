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
include("kdense_leanKAN.jl")
export KDense_lean

# Mult KAN
include("kdense_multKAN.jl")
export KDense_mult

end # module
