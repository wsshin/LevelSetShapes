export AbstractSimpleShape

abstract type AbstractSimpleShape{K} <: AbstractShape{K} end

# Below, typing x::SFloat{K} generates an error in gradient().
ndir(x::SReal{K}, s::AbstractSimpleShape{K}) where {K} = normalize(gradient(x -> level(x,s), x))

function project(x::SReal{K}, s::AbstractSimpleShape{K}) where {K}
    n̂ = ndir(x, s)
    f(α) = level(x + α*n̂, s)

    α = newton(f, 0.0).sol
    pt = x + α*n̂

    return (pt=pt, ndir=n̂)
end

include("ball.jl")
include("ellipsoid.jl")
include("halfspace.jl")
