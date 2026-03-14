abstract type AbstractShape{K} end  # K: dimension of space

broadcastable(shp::AbstractShape) = Ref(shp)

center(s::AbstractShape) = s.c
max_radius(s::AbstractShape) = maximum(s.r)

in(x::AbstractVector{<:Real}, s::AbstractShape) = in(x, s, 0)
in(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = level(x,s,δr) ≤ 0

# Return the value of the level set function of the shape.
level(x::AbstractVector{<:Real}, s::AbstractShape) = level(x, s, 0)
level(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = level(SVector{K}(x), s, δr)

level(x₁::Real, s::AbstractShape{1}, δr::Real) = level(SVector{1}(x₁), s, δr)
level(x₁::Real, x₂::Real, s::AbstractShape{2}, δr::Real) = level(SVector{2}(x₁,x₂), s, δr)
level(x₁::Real, x₂::Real, x₃::Real, s::AbstractShape{3}, δr::Real) = level(SVector{3}(x₁,x₂,x₃), s, δr)

# Below, typing x::SVector{K,Float64} generates an error in gradient().
outnormal(x::AbstractVector{<:Real}, s::AbstractShape) = outnormal(x, s, 0)
outnormal(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = outnormal(SVector{K}(x), s, δr)
outnormal(x::SVector{K,<:Real}, s::AbstractShape{K}, δr::Real) where {K} = (ForwardDiff.gradient(x -> level(x,s,δr), x))  # assume lever(...) is signed distance function

project(x::AbstractVector{<:Real}, s::AbstractShape) = project(x, s, 0)
project(x::AbstractVector{<:Real}, s::AbstractShape{K}, δr::Real) where {K} = project(SVector{K}(x), s, δr)
function project(x::SVector{K,<:Real}, s::AbstractShape{K}, δr::Real) where {K}
    n̂ = outnormal(x, s, δr)
    d = level(x, s, δr)

    pt = x - d * n̂

    return (pt=pt, ndir=n̂)
end

# Return the center position of the shape, where the level set function is minimized.
function center end

function bounds(s::AbstractShape{K}, δr::Real) where {K}
    bₙₚ = (MVector{K,Float64}(undef), MVector{K,Float64}(undef))

    c = center(s)
    # m_max = round(Int, log2(max_radius(s) / δr))
    # ms = range(m_max, 0, m_max+1)  # decreasing
    # ms = range(m_max, 0, 2)  # decreasing
    ms = 0:0
    for k in 1:K
        eₖ = SVector{K}(k′==k for k′ in 1:K)

        for (ind_parity, parity) in enumerate((-1,1))
            local x_bound
            for m = ms
                δr_max = δr * 2^m
                ∆c = 2max_radius(s)
                x_bound = c + (parity * ∆c) * eₖ
                while true
                    retcode, x_bound = dir_bound(s, δr_max, x_bound + (parity * ∆c) * eₖ, k)
                    retcode == SciMLBase.ReturnCode.Success && break
                    ∆c *= 2
                end
            end

            # for m = Iterators.drop(range(m_max, 0, 10m_max+1), 1)
            #     δr_curr = δr * 2^m
            #     retcode, x_bound = dir_bound(s, δr_curr, x_bound, k)
            #     if retcode != SciMLBase.ReturnCode.Success
            #         error("Bounds calculation failed: direction = $k, parity = $parity")
            #     end
            # end

            bₙₚ[ind_parity][k] = x_bound[k]
        end
    end

    return bₙₚ
end

function dir_bound(s::AbstractShape{K}, δr::Real, x_guess::SVector{K,Float64}, k::Integer) where {K}
    # Define the nonlinear function to solve (= to find the root).
    #
    # At the root of f(x, k) as a function of x with a parameter k, both the projected
    # gradient along the kth dimension and the level-set function of the shape s are
    # nullified.
    #
    # The projection dimension k is passed to NonlinearProblem as the paramter p of
    # NonlinearProblem.
    ∇proj_level(x, k) = projected_gradient(x->level(x,s,δr), x, k)
    f(x, p) = SVector(∇proj_level(x, p)..., level(x,s,δr))  # p = k will be used in prb

    # Define the nonlinear solution algorithm.
    #
    # SimpleNewtonRaphson(), SimpleNewtonRaphson(autodiff=AutoFiniteDiff()), SimpleBroyden(),
    # TrustedRegion() have been tested in addition to the algorithm used below.
    alg = NewtonRaphson(; linsolve=LinearSolveFunction(tsvd_solver))

    # Define parameters for constructing the initial guess solution.
    prb = NonlinearProblem{false}(f, x_guess, k)
    sol = solve(prb, alg)

    return sol.retcode, sol.u
end

include("sphere.jl")
# include("ellipsoid.jl")
include("polyhedron.jl")
