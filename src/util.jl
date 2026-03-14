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
