module LevelSetShapes

using ForwardDiff
using GoldfarbIdnaniSolver: solveQP
using LinearAlgebra
using LinearSolve
using NonlinearSolve
using StaticArrays

import Base: ==, isapprox, hash, broadcastable, in

# shape.jl
export AbstractShape
export level, outnormal, project, center, bounds

# sphere.jl
export Sphere

# ellipsoid.jl
export Ellipsoid

# polyhedron.jl
export Polyhedron

# translation.jl
export TranslatedShape

# boolean.jl
export IntersectionShape

include("util.jl")
include("shape/shape.jl")
include("operation/operation.jl")

end # module