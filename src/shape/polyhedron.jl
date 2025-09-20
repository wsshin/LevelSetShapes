struct Polyhedron{K,F,KF} <: AbstractShape{K}  # F: number of faces; V: number of vertices; KF = K⋅F
    N::SMatrix{K,F,Float64,KF}  # each column is outward unit normal to face
    r::SVector{F,Float64}  # Nᵀx ≤ r define polyhedron; nᵢᵀ x ≤ rᵢ is half space
end

function Polyhedron(N::AbstractMatrix{<:Real}, r::AbstractVector{<:Real})
    K, F = size(N)
    d = .√sum(abs2, N, dims=1)

    return Polyhedron{K,F,K*F}(N ./ d, r)
end

function level(x::SVector{K,<:Real}, s::Polyhedron{K}, δr::Real) where {K}
    Q = SMatrix{K,K,Float64}(2I)
    c = float(2x)

    # Below, A and b are negated because solveQP(Q, c, A, b) assumes that constraints are
    # Aᵀ r ≥ b instead of Aᵀ r ≤ b.
    #
    # In constructing b, s.r is reduced by δr.
    # - Near a face, the retraction increases the distance by δr, so decreasing it back by 
    # δr later recovers the distance to the original shape.
    # - Near a vertex, the retraction increases the distance by more than δr, so increaning
    # in back by δr later produces a distance greater than the distance to the original 
    # vertex: it produces the distance to a shape that is rounded at the vertex.
    #
    # Suppose we collect all the points that are δr-away from the shape whose faces are
    # retracted by δr.  Around a vertex, the radius of rounded vertex becomes exactly δr.
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

    return norm(y - x) - δr  # distance was overestimated by retraction, so reduce it by δr
end