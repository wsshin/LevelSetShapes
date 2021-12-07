export AbstractShape
export level, ndir, project, center, bounds

abstract type AbstractShape{K} end

Base.in(x::AbsVecReal, s::AbstractShape{K}) where {K} = level(x,s) ≤ 0

# Return the value of the level set function of the shape.
level(x::AbsVecReal, s::AbstractShape{K}) where {K} = level(SVec{K}(x), s)

# Return the center position of the shape, where the level set function is minimized.
function center end

# Return the minimum axis-aligned bounding box of the shape.
function bounds(s::AbstractShape{K}) where {K}
    c = center(s)
    δc = τᵣ*abs.(c) .+ τₐ  # SFloat{K}; all-positive entries

    # Calculate the negative-end corner of the axis-aligned bounding box.
    bₙ = SVec(ntuple(k->
        begin
            c′ = c - δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards negative direction of k-axis
            # c′ = c - δc
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # Calculate the positive-end corner of the axis-aligned bounding box.
    bₚ = SVec(ntuple(k->
        begin
            c′ = c + δc[k] * SVec(ntuple(k′->(k′==k), Val(K)))  # SFloat{K}; perturb c towards positive direction of k-axis
            # c′ = c + δc
            bₖ = lagrange(x->x[k], x->level(x,s), c′, rtol=τₐ).sol[k]  # Float
        end, Val(K)))

    # return c + τᵣ*(bₙ-c), c + τᵣ*(bₚ-c)  # make box slightly larger
    return bₙ, bₚ
end

# Return the outward normal direction, even for x inside the shape.
ndir(x::AbsVecReal, s::AbstractShape{K}) where {K} = ndir(SVec{K}(x), s)

# Project x onto s along the direction normal to the shape.  The result is only
# approximate.
project(x::AbsVecReal, s::AbstractShape{K}) where {K} = project(SVec{K}(x), s)

include("translated.jl")
include("simple/simple.jl")
include("composite/composite.jl")
