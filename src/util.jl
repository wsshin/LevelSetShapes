"""
    solveQP(dmat, dvec, Amat, bvec)

Solve the strictly convex quadratic program

    minimize  -dvec' * x + 1/2 * x' * dmat * x
    subject to transpose(Amat) * x ≥ bvec

This implementation is intentionally restricted to `StaticArrays`, matching how
`LevelSetShapes` constructs the QPs used by `Polyhedron`. It supports only the
inequality form used in `sdf_out(...)` and works with automatic-differentiation
types such as `ForwardDiff.Dual`.

The return value is only the solution vector `x`.
"""
function solveQP(dmat::SMatrix{K,K,TQ}, dvec::SVector{K,Td},
                 Amat::SMatrix{K,F,TA}, bvec::SVector{F,Tb}) where {K,F,TQ<:Real,Td<:Real,TA<:Real,Tb<:Real}
    tol = _qp_tolerance(dvec, bvec)
    max_active = min(K, F)

    best_sol = nothing
    best_crval = nothing

    for nactive in 0:max_active
        _foreach_combination(1, F, nactive) do active
            candidate = _solve_qp_active_set(dmat, dvec, Amat, bvec, active, tol)
            candidate === nothing && return

            sol, crval = candidate
            if best_sol === nothing || _qp_primal(crval) < _qp_primal(best_crval)
                best_sol = sol
                best_crval = crval
            end
        end
    end

    best_sol === nothing && throw(error("constraints are inconsistent, no solution!"))
    return best_sol
end

function _solve_qp_active_set(D::SMatrix{K,K,TQ}, d::SVector{K,Td},
                              A::SMatrix{K,F,TA}, b::SVector{F,Tb},
                              active::SVector{N,Int}, tol::Float64) where {K,F,N,TQ<:Real,Td<:Real,TA<:Real,Tb<:Real}
    x = D \ d

    if N == 0
        λ = SVector{0,promote_type(TQ,Td,TA,Tb)}()
    else
        Aact = A[:, active]
        Y = D \ Aact
        schur = transpose(Aact) * Y
        rhs = b[active] - transpose(Aact) * x

        local λ
        try
            λ = schur \ rhs
        catch err
            if err isa SingularException
                return nothing
            end
            rethrow(err)
        end

        x = x + Y * λ

        for i in eachindex(λ)
            _qp_primal(λ[i]) ≥ -tol || return nothing
        end
    end

    slack = transpose(A) * x - b
    isempty(slack) || mapreduce(_qp_primal, min, slack) ≥ -tol || return nothing

    crval = dot(x, D * x) / 2 - dot(d, x)

    return x, crval
end

function _foreach_combination(f::Function, first::Int, last::Int, k::Int)
    return _foreach_combination(f, first, last, Val(k), SVector{0,Int}())
end

function _foreach_combination(f::Function, first::Int, last::Int,
                              ::Val{0}, prefix::SVector{N,Int}) where {N}
    f(prefix)
    return
end

function _foreach_combination(f::Function, first::Int, last::Int,
                              ::Val{K}, prefix::SVector{N,Int}) where {K,N}
    for i in first:(last - K + 1)
        _foreach_combination(f, i + 1, last, Val(K - 1), SVector(prefix..., i))
    end
    return
end

function _qp_tolerance(d::SVector{K,Td}, b::SVector{F,Tb}) where {K,F,Td<:Real,Tb<:Real}
    RT = Float64
    scale = max(
        one(RT),
        mapreduce(x -> abs(Float64(_qp_primal(x))), max, d; init=zero(RT)),
        mapreduce(x -> abs(Float64(_qp_primal(x))), max, b; init=zero(RT)),
    )

    return sqrt(eps(RT)) * scale
end

_qp_primal(x::Real) = x
_qp_primal(x::ForwardDiff.Dual) = _qp_primal(ForwardDiff.value(x))

# Evaluate the gradient of f but remove its kth component.  Note that the gradient of f is a
# K-dimensional vector, but the result is a (K-1)-dimensional vector.
#
# This is equivalent to constructing the gradient of f with all the partial derivatives,
# i.e., ∂f/∂xᵢ for all i, except for the partial derivative ∂f/∂xₖ that is w.r.t. the kth
# component.  Therefore, the implementation attempts to collect all the partial derivatives
# except for the kth partial derivative.
function projected_gradient(f::Any, x::SVector{K,<:Real}, k::Integer) where {K}
    x_k = x[k]

    # Construct a function that does the followings:
    # - Takes a (K-1)-dimensional point x_proj.
    # - Converts x_proj to a K-dimensional point by shifting its ith components to (i+1)th
    #   components for i ≥ k and putting x_k as the kth component.
    # - Evaluate f with the constructed K-dimensional point.
    f_proj = x_proj -> f(SVector{K}(i==k ? x_k : (i<k ? x_proj[i] : x_proj[i-1]) for i in 1:K))

    x_proj = x[SVector{K-1}(i<k ? i : i+1 for i in 1:(K-1))]  # SVector{K-1}
    return ForwardDiff.gradient(f_proj, x_proj)  # SVector{K-1}
end

function tsvd_solver(
    A, b::SVector{K}, u, p, isfresh, Pl, Pr, cachelevel;
    kwargs...
) where {K}
    Amat = hcat((A * v for v in eachcol(SMatrix{K,K}(I)))...)  # construct matrix for matrix-free operator A

    # Below, rtol=0 in pinv(...) keeps all singular values and produces very large elements
    # in solution.
    #
    # Note that atol=0 by default, which sets rtol=nϵ, so rtol=nϵ is the default value.
    return pinv(Amat) * b
end
