export Ball

struct Ball{K} <: AbstractSimpleShape{K}
    c::SVector{K,Float64}  # center of ball
    r::Float64  # radius
end

Ball(c::AbstractVector{<:Real}, r::Real) = Ball{length(c)}(c,r)

Base.:(==)(s1::Ball, s2::Ball) = s1.c==s2.c && s1.r==s2.r
Base.isapprox(s1::Ball, s2::Ball) = s1.c≈s2.c && s1.r≈s2.r
Base.hash(s::Ball, h::UInt) = hash(s.c, hash(s.r, hash(:Ball, h)))

level(x::SVector{K,<:Real}, s::Ball{K}, ∆r::Real) where {K} = norm(x-s.c) - (s.r - ∆r)  # signed distance function
center(s::Ball) = s.c
bounds(s::Ball) = (s.c .- s.r, s.c .+ s.r)
