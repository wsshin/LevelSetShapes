export newton, lagrange

newton(f, x₀::Real; kwargs...) = newton_impl(f, x₀, derivative, abs; kwargs...)
newton(f, x₀::SReal; kwargs...) = newton_impl(f, x₀, jacobian, norm; kwargs...)

function newton_impl(f, x₀,
                     jfun,  # jacobian function
                     nfun;  # norm function
                     rtol::Real=τᵣ,
                     atol::Real=τₐ,
                     maxit=100,  # maximum iterations
                     maxit_ls=20)  # maximum iterations inside the line search
    # Solve f(x) = 0 using the Newton-Armijo method.
    # x₀: initial guess
    # f: function of x

    isconverged = true  # true if solution converged; false otherwise
    α = 1e-4
    ∆max = 1 / rtol  # maximum step size
    perturb_ls = atol^0.75  # ≈ 1e-12, between sqrt(eps) and eps; some perturbation allowed in line search

    # Initialize.
    n = 0
    xₙ = x₀
    fₙ = f(xₙ)
    τf = rtol*nfun(fₙ) + atol
    rx₀ = nfun(x₀)

    # Perform the Newton method.
    while nfun(fₙ) > τf
        λ = 1.
        n_ls = 0  # line search iteration counter

        f′ₙ = jfun(f, xₙ)

        local ∆x  # see https://discourse.julialang.org/t/cumbersome-scoping-rules-for-try-catch-finally/4582
        try
            ∆x = -f′ₙ \ fₙ
        catch err
            isa(err,SingularException) && @error "For xₙ = $xₙ at n = $n, f′ₙ = $f′ₙ is singular."
            break
        end

        # Avoid too large Newton steps.
        nfun(∆x) ≤ ∆max || (isconverged = false; break)
        xₙ₊₁ = xₙ + λ*∆x
        fₙ₊₁ = f(xₙ₊₁)

        # Perform the line search to determine λ.  The stopping criterion does not
        # have perturb_ls ≈ 1e-12 on the RHS, but I guess this kind of perturbation
        # allows update in xₙ even in the situation where line search is supposed
        # to fail.
        while nfun(fₙ₊₁) ≥ (1 - α*λ) * nfun(fₙ) + perturb_ls
            λ /= 2
            xₙ₊₁ = xₙ + λ*∆x
            fₙ₊₁ = f(xₙ₊₁)
            n_ls += 1

            # Too many iteration steps in line search
            n_ls ≤ maxit_ls || (isconverged = false; break)
        end

        # Step accepted; continue the Newton method.
        xₙ = xₙ₊₁
        n += 1

        # Too many iteration steps in Newton's method.
        n ≤ maxit || (isconverged = false; break)
        # n ≤ maxit || throw(ErrorException("Newton method fails to converge in $n iteration steps."))

        fₙ = f(xₙ)
    end
    isnan(nfun(fₙ)) && @error "For xₙ = $xₙ at n = $n, f(xₙ) = $fₙ has NaN."

    return (sol=xₙ, converged=isconverged)  # named tuple
end

# function newton_impl(f, x₀,
#                      jfun,  # jacobian function
#                      nfun;  # norm function
#                      rtol::Real=τᵣ,
#                      atol::Real=τₐ,
#                      maxit=100,  # maximum iterations
#                      maxit_ls=20)  # maximum iterations inside the line search
#     # Solve f(x) = 0 using the Newton-Armijo method.
#     # x₀: initial guess
#     # f: function of x
#     K = Size(x₀)[1]
#
#     isconverged = true  # true if solution converged; false otherwise
#     α = 1e-4
#     ∆max = 1 / rtol  # maximum step size
#     perturb_ls = atol^0.75  # ≈ 1e-12, between sqrt(eps) and eps; some perturbation allowed in line search
#
#     # Initialize.
#     n = 0
#     xₙ = x₀
#     fₙ = f(xₙ)
#     τf = rtol*nfun(fₙ) + atol
#     rx₀ = nfun(x₀)
#
#     # Perform the Newton method.
#     while nfun(fₙ) > τf
#         λ = 1.
#         n_ls = 0  # line search iteration counter
#
#         f′ₙ = jfun(f, xₙ)
#         bool_nz = SVec(ntuple(k->!iszero(f′ₙ[:,k]), Val(K)))
#         M = sum(bool_nz)
#         ind_nz = SVec{M}((1:K)[bool_nz])
#
#         local ∆x  # see https://discourse.julialang.org/t/cumbersome-scoping-rules-for-try-catch-finally/4582
#         try
#             ∆x = -f′ₙ[:,ind_nz] \ fₙ
#         catch err
#             isa(err,SingularException) && @error "For xₙ = $xₙ at n = $n, f′ₙ = $f′ₙ is singular."
#             break
#         end
#
#         # Avoid too large Newton steps.
#         nfun(∆x) ≤ ∆max || (isconverged = false; break)
#         # xₙ₊₁ = xₙ + λ*∆x
#         xₙ₊₁′ = MVec(xₙ)
#         xₙ₊₁′[ind_nz] .+= λ .* ∆x
#         xₙ₊₁ = SVec(xₙ₊₁′)
#         fₙ₊₁ = f(xₙ₊₁)
#
#         # # Perform the line search to determine λ.  The stopping criterion does not
#         # # have perturb_ls ≈ 1e-12 on the RHS, but I guess this kind of perturbation
#         # # allows update in xₙ even in the situation where line search is supposed
#         # # to fail.
#         # while nfun(fₙ₊₁) ≥ (1 - α*λ) * nfun(fₙ) + perturb_ls
#         #     λ /= 2
#         #     xₙ₊₁ = xₙ + λ*∆x
#         #     fₙ₊₁ = f(xₙ₊₁)
#         #     n_ls += 1
#         #
#         #     # Too many iteration steps in line search
#         #     n_ls ≤ maxit_ls || (isconverged = false; break)
#         # end
#
#         # Step accepted; continue the Newton method.
#         xₙ = xₙ₊₁
#         n += 1
#
#         # Too many iteration steps in Newton's method.
#         n ≤ maxit || (isconverged = false; break)
#         # n ≤ maxit || throw(ErrorException("Newton method fails to converge in $n iteration steps."))
#
#         fₙ = f(xₙ)
#     end
#     isnan(nfun(fₙ)) && @error "For xₙ = $xₙ at n = $n, f(xₙ) = $fₙ has NaN."
#
#     return (sol=xₙ, converged=isconverged)  # named tuple
# end

# Need to use different initial guesses for finding the left and right boundaries.
function lagrange(f_obj, f_con, x₀::SReal{K}; kwargs...) where {K}
    ind_x = SVec(ntuple(identity, Val(K)))

    y₀ = SVec(x₀.data..., 1)
    L(y) = f_obj(y[ind_x]) - y[K+1] * f_con(y[ind_x])
    ∇L(y) = gradient(L, y)

    res = newton(∇L, y₀; kwargs...)
    yₛ = res.sol
    converged = res.converged

    xₛ = yₛ[ind_x]
    λₛ = yₛ[K+1]
    Lₛ = L(yₛ)

    return (sol=xₛ, λ=λₛ, L=Lₛ, converged=converged)
end
