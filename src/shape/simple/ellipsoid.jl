export Ellipsoid

struct Ellipsoid{K,K²} <: AbstractSimpleShape{K}
    c::SFloat{K}  # center of ellipsoid
    r::SFloat{K}  # semiaxes ("radii") in axis directions
    p::S²Float{K,K²}  # projection matrix to Ellipsoid coordinates; must be orthonormal (see surfpt_nearby)
    Ellipsoid{K,K²}(c,r,p) where {K,K²} = new(c,r,p)  # suppress default outer constructor
end

function Ellipsoid(c::AbsVecReal,  # center of ellipsoid
                   r::AbsVecReal,  # semiaxes ("radii")
                   axes::AbsMatReal=S²Float{length(c)}(I))  # columns are axes vector; assumed orthogonal
    K = length(c)
    l_ax = SVec(ntuple(k->norm(@view(axes[:,k])), Val(K)))
    p = axes' ./ l_ax

    return Ellipsoid{K,K*K}(c,r,p)
end

# Ellipsoid(s::Cuboid{K,K²}) where {K,K²} = Ellipsoid{K,K²}(s.c, (s.r).^-2, s.p)

Base.:(==)(s1::Ellipsoid, s2::Ellipsoid) = s1.c==s2.c && s1.r==s2.r && s1.p==s2.p
Base.isapprox(s1::Ellipsoid, s2::Ellipsoid) = s1.c≈s2.c && s1.r≈s2.r && s1.p≈s2.p
Base.hash(s::Ellipsoid, h::UInt) = hash(s.c, hash(s.r, hash(s.p, hash(:Ellipsoid, h))))

# level(x::SReal{K}, s::Ellipsoid{K}) where {K} = norm((s.p * (x-s.c)) ./ s.r) - 1.0
function level(x::SReal{K}, s::Ellipsoid{K}) where {K}
    ∆ = (s.p * (x-s.c)) ./ s.r
    α = maximum(∆)
    return iszero(α) ? -1.0 : α*norm(∆/α)-1.0
end

center(s::Ellipsoid) = s.c
