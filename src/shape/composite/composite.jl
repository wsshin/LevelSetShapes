export CompositeShape

struct CompositeShape{K} <: AbstractShape{K}
    c::SVector{K,Float64}  # center of ball
    svec::Vector{AbstractShape{K}}
    CompositeShape{K}(c,svec) where {K} = new(c,svec)
end

CompositeShape(c::AbstractVector{<:Real}, s::AbstractShape...) = CompositeShape(c, [s...])
CompositeShape(c::AbstractVector{<:Real}, svec::AbsVec{<:AbstractShape}) = CompositeShape{length(c)}(c,svec)

level(x::SVector{K,<:Real}, s::CompositeShape{K}) where {K} = mapreduce(sᵢ->level(x,sᵢ), max, s.svec; init=-Inf)
center(s::CompositeShape) = s.c
