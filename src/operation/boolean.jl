struct IntersectionShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    shps::Vector{S}

    IntersectionShape{K}(svec) where {K} = new(shps)
end

CompositeShape(c::AbstractVector{<:Real}, s::AbstractShape...) = CompositeShape(c, [s...])
CompositeShape(c::AbstractVector{<:Real}, svec::AbsVec{<:AbstractShape}) = CompositeShape{length(c)}(c,svec)

level(x::SVector{K,<:Real}, s::CompositeShape{K}) where {K} = mapreduce(sᵢ->level(x,sᵢ), max, s.svec; init=-Inf)
center(s::CompositeShape) = s.c
