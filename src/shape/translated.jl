mutable struct TranslatedShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    s::S
    ∆::SVector{K,Float64}
end

translate(s::AbstractShape{K}, ∆::AbstractVector{<:Real}) where {K} = translate(s, SVector{K,Float64}(∆))
translate(s::AbstractShape{K}, ∆::SVector{K,Float64}) where {K} = TranslatedShape(s, ∆)
translate(s::TranslatedShape{K}, ∆::SVector{K,Float64}) where {K} = TranslatedShape(s, s.∆+∆)

level(x::SVector{K,Float64}, s::TranslatedShape{K}) where {K} = level(x-s.∆, s.s)
project(x::SVector{K,Float64}, s::TranslatedShape{K}) where {K} = project(x-s.∆, s.s)
ndir(x::SVector{K,Float64}, s::TranslatedShape{K}) where {K} = ndir(x-s.∆, s.s) # outward direction even for x inside s
