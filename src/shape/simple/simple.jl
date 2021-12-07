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

# Return the minimum axis-aligned bounding box of the shape.
function bounds(s::AbstractSimpleShape{K}) where {K}
    c = center(s)
    δc = τᵣ*abs.(c) .+ τₐ  # SFloat{K}; all-positive entries

    # Calculate the negative-end corner of the axis-aligned bounding box.
    bₙ = SVec(ntuple(k->
        begin
            c′ = c - δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards negative direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # Calculate the positive-end corner of the axis-aligned bounding box.
    bₚ = SVec(ntuple(k->
        begin
            c′ = c + δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards positive direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # return c + τᵣ*(bₙ-c), c + τᵣ*(bₚ-c)  # make box slightly larger
    return bₙ, bₚ
end

include("ball.jl")
include("ellipsoid.jl")
include("halfspace.jl")
