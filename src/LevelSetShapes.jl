module LevelSetShapes

using ForwardDiff: derivative, gradient, jacobian
using LinearAlgebra
using StaticArrays

include("shape/shape.jl")
include("operation/operation.jl")

end # module
