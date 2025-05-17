export AbstractSimpleShape

abstract type AbstractSimpleShape{K} <: AbstractShape{K} end

# Below, typing x::SVector{K,Float64} generates an error in gradient().
ndir(x::SVector{K,<:Real}, s::AbstractSimpleShape{K}, ∆r::Real=0) where {K} = gradient(x -> level(x,s,∆r), x)  # assume lever(...) is signed distance function

function project(x::SVector{K,<:Real}, s::AbstractSimpleShape{K}, ∆r::Real=0) where {K}
    n̂ = ndir(x, s)
    d = level(x, s, ∆r)

    pt = x - d*n̂

    return (pt=pt, ndir=n̂)
end

include("ball.jl")
# include("ellipsoid.jl")
# include("halfspace.jl")
