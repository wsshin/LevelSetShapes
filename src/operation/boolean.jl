struct IntersectionShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    shps::Vector{S}
end

IntersectionShape(shps::AbstractShape...) = IntersectionShape(SVector(shps))
IntersectionShape(shps::AbstractVector{<:AbstractShape}) = IntersectionShape{length(shps),eltype(shps)}(shps)

level(x::SVector{K,<:Real}, shp::IntersectionShape{K}) where {K} = mapreduce(shpᵢ->level(x,shpᵢ), max, shp.shps; init=-Inf)
