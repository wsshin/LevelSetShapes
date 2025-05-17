export Ellipsoid, Ball

struct Ellipsoid{K,K²} <: AbstractSimpleShape{K}
    c::SVector{K,Float64}  # center of ellipsoid
    r::SVector{K,Float64}  # semiaxes ("radii") in axis directions
    p::S²Float{K,K²}  # projection matrix to Ellipsoid coordinates; must be orthonormal (see surfpt_nearby)
    Ellipsoid{K,K²}(c,r,p) where {K,K²} = new(c,r,p)  # suppress default outer constructor
end

function Ellipsoid(c::AbstractVector{<:Real},  # center of ellipsoid
                   r::AbstractVector{<:Real},  # semiaxes ("radii")
                   axes::AbsMatReal=S²Float{length(c)}(I))  # columns are axes vector; assumed orthogonal
    K = length(c)
    l_ax = SVector(ntuple(k->norm(@view(axes[:,k])), Val(K)))
    p = axes' ./ l_ax

    return Ellipsoid{K,K*K}(c,r,p)
end

Ball(c::AbsVecReal, r::Real) = (K = length(c); Ellipsoid(c, @SVector(fill(r,K))))

Base.:(==)(s1::Ellipsoid, s2::Ellipsoid) = s1.c==s2.c && s1.r==s2.r && s1.p==s2.p
Base.isapprox(s1::Ellipsoid, s2::Ellipsoid) = s1.c≈s2.c && s1.r≈s2.r && s1.p≈s2.p
Base.hash(s::Ellipsoid, h::UInt) = hash(s.c, hash(s.r, hash(s.p, hash(:Ellipsoid, h))))

level(x::SVector{K,<:Real}, s::Ellipsoid{K}) where {K} = norm((s.p * (x-s.c)) ./ s.r) - 1.0
center(s::Ellipsoid) = s.c
