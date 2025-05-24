struct IntersectionShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    shps::Vector{S}

    IntersectionShape{K}(shps) where {K} = new(shps)
end

CompositeShape(shps::AbstractShape...) = CompositeShape(SVector(shps))
CompositeShape(shps::AbstractVector{<:AbstractShape}) = CompositeShape{length(shps)}(shps)

level(x::SVector{K,<:Real}, shp::CompositeShape{K}) where {K} = mapreduce(shpᵢ->level(x,shpᵢ), max, shp.shps; init=-Inf)
