struct Sphere{K} <: AbstractShape{K}
    c::SVector{K,Float64}  # center of sphere
    r::Float64  # radius
end

Sphere(c::AbstractVector{<:Real}, r::Real) = Sphere{length(c)}(c,r)

(==)(s1::Sphere, s2::Sphere) = s1.c==s2.c && s1.r==s2.r
isapprox(s1::Sphere, s2::Sphere) = s1.c≈s2.c && s1.r≈s2.r
hash(s::Sphere, h::UInt) = hash(s.c, hash(s.r, hash(:Sphere, h)))

level(x::SVector{K,<:Real}, s::Sphere{K}, δr::Real) where {K} = norm(x-s.c) - s.r  # signed distance function
bounds(s::Sphere) = (s.c .- s.r, s.c .+ s.r)
