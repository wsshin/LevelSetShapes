export HalfSpace

struct HalfSpace{K} <: AbstractShape{K}
    x₀::SVector{K,Float64}  # point on boundary
    n̂::SVector{K,Float64}  # outward direction normal
end

HalfSpace(x₀::AbstractVector{<:Real}, n::AbstractVector{<:Real}) = HalfSpace{length(x₀)}(x₀, normalize(n))

Base.:(==)(s1::HalfSpace, s2::HalfSpace) = s1.x₀==s2.x₀ && s1.n̂==s2.n̂
Base.isapprox(s1::HalfSpace, s2::HalfSpace) = s1.x₀≈s2.x₀ && s1.n̂≈s2.n̂
Base.hash(s::HalfSpace, h::UInt) = hash(s.x₀, hash(s.n̂, hash(:HalfSpace, h)))

level(x::SVector{K,<:Real}, s::HalfSpace{K}) where {K} = s.n̂ ⋅ (x-s.x₀)  # signed distance function

# The following functions cannot be implemented for HalfSpace.
# center()
# bounds()
