export AbstractShape
export level, ndir, project, center, bounds

abstract type AbstractShape{K} end

Base.in(x::AbstractVector{<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K} = level(x,s, ∆r) ≤ 0

# Return the value of the level set function of the shape.
level(x::AbstractVector{<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K} = level(SVector{K}(x), s, ∆r)

# Below, typing x::SVector{K,Float64} generates an error in gradient().
ndir(x::SVector{K,<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K} = gradient(x -> level(x,s,∆r), x)  # assume lever(...) is signed distance function

function project(x::SVector{K,<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K}
    n̂ = ndir(x, s)
    d = level(x, s, ∆r)

    pt = x - d * n̂

    return (pt=pt, ndir=n̂)
end

# Return the center position of the shape, where the level set function is minimized.
function center end

# Return the minimum axis-aligned bounding box of the shape.
function bounds(s::AbstractShape{K}) where {K}
    c = center(s)
    δc = τᵣ*abs.(c) .+ τₐ  # SVector{K,Float64}; all-positive entries

    # Calculate the negative-end corner of the axis-aligned bounding box.
    bₙ = SVector(ntuple(k->
        begin
            c′ = c - δc[k] * SVector(ntuple(k′->(k′==k), Val(K)))  # SVector{K,Float64}; perturb c towards negative direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float64
        end, Val(K)))

    # Calculate the positive-end corner of the axis-aligned bounding box.
    bₚ = SVector(ntuple(k->
        begin
            c′ = c + δc[k] * SVector(ntuple(k′->(k′==k), Val(K)))  # SVector{K,Float64}; perturb c towards positive direction of k-axis
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float64
        end, Val(K)))

    # return c + τᵣ*(bₙ-c), c + τᵣ*(bₚ-c)  # make box slightly larger
    return bₙ, bₚ
end

# Return the outward normal direction, even for x inside the shape.
ndir(x::AbstractVector{<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K} = ndir(SVector{K}(x), s, ∆r)

# Project x onto s along the direction normal to the shape.  The result is only
# approximate.
project(x::AbstractVector{<:Real}, s::AbstractShape{K}, ∆r::Real=0) where {K} = project(SVector{K}(x), s, ∆r)

include("ball.jl")
# include("ellipsoid.jl")
include("halfspace.jl")
