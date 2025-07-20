struct HalfSpace{K} <: AbstractShape{K}
    x₀::SVector{K,Float64}  # point on boundary
    n̂::SVector{K,Float64}  # outward direction normal
end

HalfSpace(x₀::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = HalfSpace{length(x₀)}(x₀, normalize(n))

(==)(s1::HalfSpace, s2::HalfSpace) = s1.x₀==s2.x₀ && s1.n̂==s2.n̂
isapprox(s1::HalfSpace, s2::HalfSpace) = s1.x₀≈s2.x₀ && s1.n̂≈s2.n̂
hash(s::HalfSpace, h::UInt) = hash(s.x₀, hash(s.n̂, hash(:HalfSpace, h)))

level(x::SVector{K,<:Real}, s::HalfSpace{K}) where {K} = s.n̂ ⋅ (x-s.x₀)  # signed distance function

# The following functions cannot be implemented for HalfSpace.
# center()
# bounds()

struct Polyhedron{K,F,KF} <: AbstractShape{K}  # F: number of faces; V: number of vertices; KF = K⋅F
    N::SMatrix{K,F,Float64,KF}  # each column is outward unit normal to face
    r::SVector{F,Float64}  # Nᵀx ≤ r define polyhedron; nᵢᵀ x ≤ rᵢ is half space
end

Polyhedron(N::AbstractMatrix{<:Real}, r::AbstractVector{<:Real}) = Polyhedron{size(N)...,length(N)}(N, r)

function level(x::SVector{K,<:Real}, s::Polyhedron{K}, δr::Real) where {K}
    Q = SMatrix{K,K,Float64}(2I)
    c = float(2x)

    # Below, A and b are negated because solveQP(Q, c, A, b) assumes that constraints are
    # Aᵀ r ≥ b instead of Aᵀ r ≤ b.
    A = -s.N
    b = -(s.r .- δr)  # calculate SDF for sharp shape retreated by δr

    # The following solveQP(...) minimizes f(y) = -cᵀ y + ½ yᵀ Q y, subject to Aᵀ y ≥ b
    # (rather than Aᵀ y = b, as the keyword argument `meq` specifying the number of equality
    # constraints is set to 0 by default).  Here, y represents a point in the the polyhedron.
    #
    # For a given point x, we want to find y minimizing ‖y - x‖² = yᵀy - 2 xᵀy + xᵀx, or y
    # minimizing yᵀy - 2 xᵀy, equivalently.  For Q = 2I and c = 2x, f(y) = -2 xᵀy + yᵀy,
    # which is exactly the function we want to minimize.
    #
    # Note that when the polygon changes, Q and c do not change; only A and b change.
    y = solveQP(Q, c, A, b)[1]  # solveQP(...) returns sol, lagr, crval, iact, nact, iter

    return norm(y - x) - δr  # distance was overestimated by retraction, so reduce by δr
end