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

# Return the minimum axis-aligned bounding box of the shape.
function bounds(s::AbstractShape{K}, δr::Real) where {K}
    # Define the nonlinear function to solve (= to find the root).
    ∇proj_level(x, k) = projected_gradient(x->level(x,s,δr), x, k, Val(K))
    f(x, p) = SVector(∇proj_level(x, p)..., level(x,s,δr))  # p = k will be used in prb

    # Define the nonlinear solution algorithm.
    #
    # SimpleNewtonRaphson(), SimpleNewtonRaphson(autodiff=AutoFiniteDiff()), SimpleBroyden()
    # have been tested in addition to the algorithm used below.
    alg = NewtonRaphson(; linsolve=LinearSolveFunction(tsvd_solver))
    kwargs_sol = (; )

    # Define parameters for constructing the initial guess solution.
    c = center(s)
    ∆c = 2max_radius(s)
    # ∆c = (1+Base.rtoldefault(Float64)) * max_radius(s)
    # ∆c = 1e-1max_radius(s)
    # ∆c = 3δr
    # ∆c = min_radius(s) * Base.rtoldefault(Float64)

    # Calculeate the negative-end corner of the axis-aligned bounding box.
    bₙ = SVector(ntuple(k->
        begin
            # c′ = c .- ∆c
            c′ = c - ∆c * SVector(ntuple(k′->(k′==k), Val(K)))
            # println()
            # println("c′ for bₙ for k = $k: $c′")
            prb = NonlinearProblem{false}(f, c′, k)
            sol = solve(prb, alg; kwargs_sol...)
            sol.u[k]
        end,
        Val(K))
    )

    # Calculate the positive-end corner of the axis-aligned bounding box.
    bₚ = SVector(ntuple(k->
        begin
            # c′ = c .+ ∆c
            c′ = c + ∆c * SVector(ntuple(k′->(k′==k), Val(K)))
            # println()
            # println("c′ for bₚ for k = $k: $c′")
            prb = NonlinearProblem{false}(f, c′, k)
            sol = solve(prb, alg; kwargs_sol...)
            sol.u[k]
        end,
        Val(K))
    )

    return bₙ, bₚ
end

include("sphere.jl")
# include("ellipsoid.jl")
include("polyhedron.jl")
