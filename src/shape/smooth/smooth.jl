export AbstractSmoothShape

abstract type AbstractSmoothShape{K} <: AbstractShape{K} end

# Below, typing x::SFloat{K} generates an error in gradient().
ndir(x::SReal{K}, s::AbstractSmoothShape{K}) where {K} = normalize(gradient(x -> level(x,s), x))

function project(x::SReal{K}, s::AbstractSmoothShape{K}) where {K}
    n̂ = ndir(x, s)
    f(α) = level(x + α*n̂, s)

    α = newton(f, 0.0).sol
    xₚ = x + α*n̂

    return (pt=xₚ, ndir=n̂)
end

# Return the minimum axis-aligned bounding box of the shape.
function bounds(s::AbstractSmoothShape{K}) where {K}
    c = center(s)
    δc = τᵣ*abs.(c) .+ τₐ  # SFloat{K}; all-positive entries

    # Calculate the negative-end corner of the axis-aligned bounding box.
    b₋ = SVec(ntuple(k->
        begin
            c′ = c - δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards negative direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # Calculate the positive-end corner of the axis-aligned bounding box.
    b₊ = SVec(ntuple(k->
        begin
            c′ = c + δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards positive direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # return c + τᵣ*(b₋-c), c + τᵣ*(b₊-c)  # make box slightly larger
    return b₋, b₊
end

include("ellipsoid.jl")
include("halfspace.jl")
# To-do: implement an arbitrary curved shape; not sure if bounds() would work for nonconvex smooth shapes
