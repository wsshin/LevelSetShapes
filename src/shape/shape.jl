abstract type AbstractShape{K} end

broadcastable(shp::AbstractShape) = Ref(shp)

in(x, s) = in(x, s, 0)
in(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = level(x,s,δr) ≤ 0

# Return the value of the level set function of the shape.
level(x, s) = level(x, s, 0)
level(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = level(SVector{K}(x), s, δr)

level(x₁::Real, s::AbstractShape{1}, δr::Real) = level(SVector{1}(x₁), s, δr)
level(x₁::Real, x₂::Real, s::AbstractShape{2}, δr::Real) = level(SVector{2}(x₁,x₂), s, δr)
level(x₁::Real, x₂::Real, x₃::Real, s::AbstractShape{3}, δr::Real) = level(SVector{3}(x₁,x₂,x₃), s, δr)

# Below, typing x::SVector{K,Float64} generates an error in gradient().
outnormal(x, s) = outnormal(x, s, 0)
outnormal(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = outnormal(SVector{K}(x), s, δr)
outnormal(x::SVector{K,<:Real}, s::AbstractShape{K}, δr::Real) where {K} = (gradient(x -> level(x,s,δr), x))  # assume lever(...) is signed distance function

project(x, s) = project(x, s, 0)
project(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = project(SVector{K}(x), s, δr)
function project(x::SVector{K,<:Real}, s::AbstractShape{K}, δr::Real) where {K}
    n̂ = outnormal(x, s, δr)
    d = level(x, s, δr)

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

include("sphere.jl")
# include("ellipsoid.jl")
include("polyhedron.jl")
