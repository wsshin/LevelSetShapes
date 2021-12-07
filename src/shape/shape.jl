export AbstractShape
export level, ndir, project, center, bounds

abstract type AbstractShape{K} end

Base.in(x::AbsVecReal, s::AbstractShape{K}) where {K} = level(x,s) ≤ 0

# Return the value of the level set function of the shape.
level(x::AbsVecReal, s::AbstractShape{K}) where {K} = level(SVec{K}(x), s)

# Return the center position of the shape, where the level set function is minimized.
function center end

# Return the minimum axis-aligned bounding box of the shape.
function bounds end

# Return the outward normal direction, even for x inside the shape.
ndir(x::AbsVecReal, s::AbstractShape{K}) where {K} = ndir(SVec{K}(x), s)

# Project x onto s along the direction normal to the shape.  The result is only
# approximate.
project(x::AbsVecReal, s::AbstractShape{K}) where {K} = project(SVec{K}(x), s)

include("translated.jl")
include("simple/simple.jl")
include("composite/composite.jl")
