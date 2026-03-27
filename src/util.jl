"""
    solveQP(dmat, dvec, Amat, bvec)

Solve the strictly convex quadratic program

    minimize  -dvec' * x + 1/2 * x' * dmat * x
    subject to transpose(Amat) * x ≥ bvec

This implementation is intentionally restricted to `StaticArrays`, matching how
`LevelSetShapes` constructs the QPs used by `Polyhedron`. It supports only the
inequality form used in `sdf_out(...)` and returns only the solution vector
`x`.

Internally it uses a narrowed local port of the Goldfarb-Idnani active-set
method, specialized to:
  * inequality constraints only
  * unfactorized `D`
  * dense small problems
"""
function solveQP(dmat::SMatrix{K,K,TQ}, dvec::SVector{K,Td},
                 Amat::SMatrix{K,F,TA}, bvec::SVector{F,Tb}) where {K,F,TQ<:Real,Td<:Real,TA<:Real,Tb<:Real}
    T = promote_type(TQ, Td, TA, Tb)
    D = Matrix{T}(dmat)
    d = Vector{T}(dvec)
    A = Matrix{T}(Amat)
    b = Vector{T}(bvec)

    return SVector{K,T}(Tuple(_solve_qp_goldfarb_idnani!(D, d, A, b)))
end

function _solve_qp_goldfarb_idnani!(dmat::Matrix{T}, dvec::Vector{T},
                                    amat::Matrix{T}, bvec::Vector{T}) where {T<:Real}
    n = size(dmat, 1)
    q = size(amat, 2)

    n == size(dmat, 2) || throw(error("Dmat is not symmetric!"))
    n == length(dvec) || throw(error("Dmat and dvec are incompatible!"))
    n == size(amat, 1) || throw(error("Incorrect number of rows. Amat amd dvec are incompatible!"))
    q == length(bvec) || throw(error("Incorrect number of columns. Amat, bvec incompatible!"))

    r = min(n, q)
    work = zeros(T, 2n + trunc(Int, r * (r + 5) / 2) + 2q + 1)
    sol = zeros(T, n)
    iact = zeros(Int, q)
    vsmall = 2 * eps(real(T))

    work[1:n] = dvec
    for i = (n + 1):length(work)
        work[i] = zero(T)
    end
    for i = 1:q
        iact[i] = 0
    end

    try
        R = cholesky!(Symmetric(dmat)).U
        ldiv!(dvec, R, ldiv!(similar(dvec), R', dvec))
        dmat .= triu!(inv(R))
    catch err
        if err isa PosDefException
            throw(error("matrix D in quadratic function is not positive definite!"))
        end
        rethrow(err)
    end

    crval = zero(T)
    for j = 1:n
        sol[j] = dvec[j]
        crval += work[j] * sol[j]
        work[j] = zero(T)
        for i = (j + 1):n
            dmat[i, j] = zero(T)
        end
    end
    crval = -crval / 2

    iwzv = n
    iwrv = iwzv + n
    iwuv = iwrv + r
    iwrm = iwuv + r + 1
    iwsv = iwrm + trunc(Int, r * (r + 1) / 2)
    iwnbv = iwsv + q

    for i = 1:q
        sum = zero(T)
        for j = 1:n
            sum += amat[j, i]^2
        end
        work[iwnbv + i] = sqrt(sum)
        isnan(sum) && throw(DomainError(sum))
    end

    iter2 = 0
    nact = 0
    local it1::Int, l::Int, l1::Int, nvl::Int
    local t1inf::Bool, t2min::Bool
    local sum::T, temp::T, t1::T, tt::T, gc::T, gs::T, nu::T

    @label L50
    l = iwsv
    for i = 1:q
        l += 1
        sum = -bvec[i]
        for j = 1:n
            sum += amat[j, i] * sol[j]
        end
        if abs(sum) < vsmall
            sum = zero(T)
        end
        work[l] = sum
    end

    for i = 1:nact
        work[iwsv + iact[i]] = zero(T)
    end

    nvl = 0
    temp = zero(T)
    for i = 1:q
        if work[iwsv + i] < temp * work[iwnbv + i]
            nvl = i
            temp = work[iwsv + i] / work[iwnbv + i]
            if work[iwsv + i] == zero(T)
                temp = zero(T)
            end
        end
    end
    nvl == 0 && return sol

    @label L55
    for i = 1:n
        sum = zero(T)
        for j = 1:n
            sum += dmat[j, i] * amat[j, nvl]
        end
        isnan(sum) && throw(DomainError(sum))
        work[i] = sum
    end

    l1 = iwzv
    for i = 1:n
        work[l1 + i] = zero(T)
    end
    for j = (nact + 1):n
        for i = 1:n
            work[l1 + i] += dmat[i, j] * work[j]
        end
    end

    t1inf = true
    for i = nact:-1:1
        sum = work[i]
        l = iwrm + trunc(Int, i * (i + 3) / 2)
        l1 = l - i
        for j = (i + 1):nact
            sum -= work[l] * work[iwrv + j]
            l += j
        end
        if sum != zero(T)
            sum /= work[l1]
        end
        isnan(sum) && throw(DomainError(sum))
        work[iwrv + i] = sum
        if sum ≤ zero(T)
            continue
        end
        t1inf = false
        it1 = i
    end

    if !t1inf
        t1 = work[iwuv + it1] / work[iwrv + it1]
        if work[iwuv + it1] == zero(T)
            t1 = zero(T)
        end
        for i = 1:nact
            if work[iwrv + i] ≤ zero(T)
                continue
            end
            temp = work[iwuv + i] / work[iwrv + i]
            if work[iwuv + i] == zero(T)
                temp = zero(T)
            end
            if temp < t1
                t1 = temp
                it1 = i
            end
        end
    end

    sum = zero(T)
    for i = (iwzv + 1):(iwzv + n)
        sum += work[i]^2
    end
    if sum ≤ vsmall
        if t1inf
            throw(error("constraints are inconsistent, no solution!"))
        else
            for i = 1:nact
                work[iwuv + i] -= t1 * work[iwrv + i]
            end
            work[iwuv + nact + 1] += t1
            @goto L700
        end
    else
        sum = zero(T)
        for i = 1:n
            sum += work[iwzv + i] * amat[i, nvl]
        end
        tt = -work[iwsv + nvl] / sum
        if work[iwsv + nvl] == zero(T)
            tt = zero(T)
        end
        t2min = true
        if !t1inf && t1 < tt
            tt = t1
            t2min = false
        end

        for i = 1:n
            sol[i] += tt * work[iwzv + i]
        end
        crval += tt * sum * (tt / 2 + work[iwuv + nact + 1])
        for i = 1:nact
            work[iwuv + i] -= tt * work[iwrv + i]
        end
        work[iwuv + nact + 1] += tt

        if t2min
            nact += 1
            iact[nact] = nvl

            l = iwrm + trunc(Int, (nact - 1) * nact / 2) + 1
            for i = 1:(nact - 1)
                work[l] = work[i]
                l += 1
            end

            if nact == n
                work[l] = work[n]
            else
                for i = n:-1:(nact + 1)
                    if work[i] == zero(T)
                        continue
                    end
                    gc = max(abs(work[i - 1]), abs(work[i]))
                    gs = min(abs(work[i - 1]), abs(work[i]))
                    temp = copysign(max(gc, sqrt(gs^2 + gc^2)), work[i - 1])
                    isnan(temp) && throw(DomainError(temp))

                    if work[i - 1] == zero(T)
                        gc = zero(T)
                    else
                        gc = work[i - 1] / temp
                    end

                    if work[i] == zero(T)
                        gs = zero(T)
                    else
                        gs = work[i] / temp
                    end

                    if gc == one(T)
                        continue
                    end
                    if gc == zero(T)
                        work[i - 1] = temp * sign(gs)
                        for j = 1:n
                            temp = dmat[j, i - 1]
                            dmat[j, i - 1] = dmat[j, i]
                            dmat[j, i] = temp
                        end
                    else
                        work[i - 1] = temp
                        nu = gs / (one(T) + gc)
                        for j = 1:n
                            temp = gc * dmat[j, i - 1] + gs * dmat[j, i]
                            dmat[j, i] = nu * (dmat[j, i - 1] + temp) - dmat[j, i]
                            dmat[j, i - 1] = temp
                        end
                    end
                end
                work[l] = work[nact]
            end
        else
            sum = -bvec[nvl]
            for j = 1:n
                sum += sol[j] * amat[j, nvl]
            end
            work[iwsv + nvl] = sum
            @goto L700
        end
    end
    @goto L50

    @label L700
    if it1 == nact
        @goto L799
    end

    @label L797
    l = iwrm + trunc(Int, it1 * (it1 + 1) / 2) + 1
    l1 = l + it1
    if work[l1] == zero(T)
        @goto L798
    end
    gc = max(abs(work[l1 - 1]), abs(work[l1]))
    gs = min(abs(work[l1 - 1]), abs(work[l1]))
    temp = copysign(max(gc, sqrt(gs^2 + gc^2)), work[l1 - 1])

    if work[l1 - 1] == zero(T)
        gc = zero(T)
    else
        gc = work[l1 - 1] / temp
    end

    if work[l1] == zero(T)
        gs = zero(T)
    else
        gs = work[l1] / temp
    end
    (isnan(gs) || isnan(gc) || isnan(temp)) && throw(DomainError(temp))

    if gc == one(T)
        @goto L798
    end
    if gc == zero(T)
        for i = (it1 + 1):nact
            temp = work[l1 - 1]
            work[l1 - 1] = work[l1]
            work[l1] = temp
            l1 += i
        end
        for i = 1:n
            temp = dmat[i, it1]
            dmat[i, it1] = dmat[i, it1 + 1]
            dmat[i, it1 + 1] = temp
        end
    else
        nu = gs / (one(T) + gc)
        if gs == zero(T)
            nu = zero(T)
        end
        for i = (it1 + 1):nact
            temp = gc * work[l1 - 1] + gs * work[l1]
            work[l1] = nu * (work[l1 - 1] + temp) - work[l1]
            work[l1 - 1] = temp
            l1 += i
        end
        for i = 1:n
            temp = gc * dmat[i, it1] + gs * dmat[i, it1 + 1]
            dmat[i, it1 + 1] = nu * (dmat[i, it1] + temp) - dmat[i, it1 + 1]
            dmat[i, it1] = temp
        end
    end

    @label L798
    l1 = l - it1
    for i = 1:it1
        work[l1] = work[l]
        l += 1
        l1 += 1
    end

    work[iwuv + it1] = work[iwuv + it1 + 1]
    iact[it1] = iact[it1 + 1]
    it1 += 1
    if it1 < nact
        @goto L797
    end

    @label L799
    work[iwuv + nact] = work[iwuv + nact + 1]
    work[iwuv + nact + 1] = zero(T)
    iact[nact] = 0
    nact -= 1
    iter2 += 1
    @goto L55
end

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
