export HalfSpace

struct HalfSpace{K} <: AbstractSimpleShape{K}
    x₀::SFloat{K}  # point on boundary
    n̂::SFloat{K}  # outward direction normal
    HalfSpace{K}(x₀,n̂) where {K} = new(x₀,n̂)  # suppress default outer constructor
end

HalfSpace(x₀::AbsVecReal, n::AbsVecReal) = HalfSpace{length(x₀)}(x₀, normalize(n))

Base.:(==)(s1::HalfSpace, s2::HalfSpace) = s1.x₀==s2.x₀ && s1.n̂==s2.n̂
Base.isapprox(s1::HalfSpace, s2::HalfSpace) = s1.x₀≈s2.x₀ && s1.n̂≈s2.n̂
Base.hash(s::HalfSpace, h::UInt) = hash(s.x₀, hash(s.n̂, hash(:HalfSpace, h)))

level(x::SReal{K}, s::HalfSpace{K}) where {K} = s.n̂ ⋅ (x-s.x₀)

# The following functions cannot be implemented for HalfSpace.
# center()
# bounds()
