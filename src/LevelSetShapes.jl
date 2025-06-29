module LevelSetShapes

using ForwardDiff: derivative, gradient, jacobian
using LinearAlgebra
using StaticArrays

import Base: ==, isapprox, hash, broadcastable, in

# shape.jl
export AbstractShape
export level, outnormal, project, center, bounds

# ball.jl
export Ball

# ellipsoid.jl
export Ellipsoid

# halfspace.jl
export HalfSpace

# translation.jl
export TranslatedShape

# boolean.jl
export IntersectionShape

include("shape/shape.jl")
include("operation/operation.jl")

end # module
