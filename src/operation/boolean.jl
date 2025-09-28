struct IntersectionShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    shps::Vector{S}
end

IntersectionShape(shps::AbstractShape...) = IntersectionShape(SVector(shps))
IntersectionShape(shps::AbstractVector{<:AbstractShape}) = IntersectionShape{length(shps),eltype(shps)}(shps)

# # This was the first working level(...) that is capable of rounding corners, but it was non-
# # differentiable.
# function level(x::SVector{K,<:Real}, shp::IntersectionShape{K}, δr::Real=0) where {K}
#     lv_sharp = mapreduce(shpᵢ->level(x,shpᵢ), max, shp.shps; init=-Inf)
#     if lv_sharp < -δr
#         lv_round = lv_sharp
#     else
#         lv_round = zero(lv_sharp)
#         for shpᵢ = shp.shps
#             lvᵢ_recess = level(x, shpᵢ) + δr
#             dᵢ = max(0, lvᵢ_recess)
#             lv_round += dᵢ^2
#         end
#         lv_round = √lv_round - δr
#     end

#     return lv_round
# end

# # This is the updated version that is differentiable.
# level(x::SVector{K,<:Real}, shp::IntersectionShape{K}, δr::Real=0) where {K} = 
#     mapreduce(shpᵢ->level(x,shpᵢ,δr), max, shp.shps; init=-Inf)