mutable struct TranslatedShape{K,S<:AbstractShape{K}} <: AbstractShape{K}
    s::S
    ∆::SFloat{K}
end

translate(s::AbstractShape{K}, ∆::AbsVecReal) where {K} = translate(s, SFloat{K}(∆))
translate(s::AbstractShape{K}, ∆::SFloat{K}) where {K} = TranslatedShape(s, ∆)
translate(s::TranslatedShape{K}, ∆::SFloat{K}) where {K} = TranslatedShape(s, s.∆+∆)

level(x::SFloat{K}, s::TranslatedShape{K}) where {K} = level(x-s.∆, s.s)
project(x::SFloat{K}, s::TranslatedShape{K}) where {K} = project(x-s.∆, s.s)
ndir(x::SFloat{K}, s::TranslatedShape{K}) where {K} = ndir(x-s.∆, s.s) # outward direction even for x inside s
