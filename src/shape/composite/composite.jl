export CompositeShape

struct CompositeShape{K} <: AbstractShape{K}
    c::SFloat{K}  # center of ball
    svec::Vector{AbstractShape{K}}
    CompositeShape{K}(c,svec) where {K} = new(c,svec)
end

CompositeShape(c::AbsVecReal, s::AbstractShape...) = CompositeShape(c, [s...])
CompositeShape(c::AbsVecReal, svec::AbsVec{<:AbstractShape}) = CompositeShape{length(c)}(c,svec)

level(x::SReal{K}, s::CompositeShape{K}) where {K} = mapreduce(sᵢ->level(x,sᵢ), max, s.svec; init=-Inf)
center(s::CompositeShape) = s.c
