export Ball

struct Ball{K} <: AbstractSimpleShape{K}
    c::SFloat{K}  # center of ball
    r::Float  # radius
    Ball{K}(c,r) where {K} = new(c,r)  # suppress default outer constructor
end

Ball(c::AbsVecReal, r::Real) = Ball{length(c)}(c,r)

Base.:(==)(s1::Ball, s2::Ball) = s1.c==s2.c && s1.r==s2.r
Base.isapprox(s1::Ball, s2::Ball) = s1.c≈s2.c && s1.r≈s2.r
Base.hash(s::Ball, h::UInt) = hash(s.c, hash(s.r, hash(:Ball, h)))

level(x::SReal{K}, s::Ball{K}) where {K} = norm(x-s.c) / s.r - 1.0
center(s::Ball) = s.c
bounds(s::Ball) = (s.c .- s.r, s.c .+ s.r)
