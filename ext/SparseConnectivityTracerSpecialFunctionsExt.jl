module SparseConnectivityTracerSpecialFunctionsExt

import SparseConnectivityTracer as SCT
using SpecialFunctions:
    SpecialFunctions,
    airyai,
    airyaiprime,
    airyaiprimex,
    airyaix,
    airybi,
    airybiprime,
    airybiprimex,
    airybix,
    besseli,
    besselix,
    besselj,
    besselj0,
    besselj1,
    besseljx,
    besselk,
    besselkx,
    bessely,
    bessely0,
    bessely1,
    besselyx,
    beta,
    cosint,
    digamma,
    ellipe,
    ellipk,
    erf,
    erfc,
    erfcinv,
    erfcx,
    erfinv,
    expint,
    expinti,
    expintx,
    gamma,
    invdigamma,
    jinc,
    logbeta,
    logerfc,
    loggamma,
    sinint,
    sphericalbesselj,
    sphericalbessely,
    trigamma

#=
Complex functions are ignored.
Functions with more than 2 arguments are ignored.

TODO: add functions with integer arguments.
=#

## 1-to-1

# ops_1_to_1_s:
# x -> f  != 0
# ∂f/∂x   != 0
# ∂²f/∂x² != 0
ops_1_to_1_s = (
    # Gamma Function
    gamma,
    loggamma,
    digamma,
    invdigamma,
    trigamma,
    # Exponential and Trigonometric Integrals
    expinti,
    sinint,
    cosint,
    # Error functions, Dawson's and Fresnel Integrals
    erf,
    erfc,
    erfcinv,
    erfcx,
    logerfc,
    erfinv,
    # Airy and Related Functions
    airyai,
    airyaiprime,
    airybi,
    airybiprime,
    airyaix,
    airyaiprimex,
    airybix,
    airybiprimex,
    # Bessel Functions
    besselj0,
    besselj1,
    bessely0,
    bessely1,
    jinc,
    # Elliptic Integrals
    ellipk,
    ellipe,
)

for op in ops_1_to_1_s
    T = typeof(op)
    @eval SCT.is_der1_zero_global(::$T) = false
    @eval SCT.is_der2_zero_global(::$T) = false
end

ops_1_to_1 = ops_1_to_1_s

## 2-to-1

# ops_2_to_1_ssc:
# ∂f/∂x    != 0
# ∂²f/∂x²  != 0
# ∂f/∂y    != 0
# ∂²f/∂y²  != 0
# ∂²f/∂x∂y != 0
ops_2_to_1_ssc = (
    # Gamma Function
    gamma,
    loggamma,
    beta,
    logbeta,
    # Exponential and Trigonometric Integrals
    expint,
    expintx,
    # Error functions, Dawson's and Fresnel Integrals
    erf,
    # Bessel Functions
    besselj,
    besseljx,
    sphericalbesselj,
    bessely,
    besselyx,
    sphericalbessely,
    besseli,
    besselix,
    besselk,
    besselkx,
)

for op in ops_2_to_1_ssc
    T = typeof(op)
    @eval SCT.is_der1_arg1_zero_global(::$T) = false
    @eval SCT.is_der2_arg1_zero_global(::$T) = false
    @eval SCT.is_der1_arg2_zero_global(::$T) = false
    @eval SCT.is_der2_arg2_zero_global(::$T) = false
    @eval SCT.is_der_cross_zero_global(::$T) = false
end

ops_2_to_1 = ops_2_to_1_ssc

## Overloads
eval(SCT.generate_code_1_to_1(:SpecialFunctions, ops_1_to_1))
eval(SCT.generate_code_2_to_1(:SpecialFunctions, ops_2_to_1))

## List operators for later testing
SCT.test_operators_1_to_1(::Val{:SpecialFunctions}) = ops_1_to_1
SCT.test_operators_2_to_1(::Val{:SpecialFunctions}) = ops_2_to_1
SCT.test_operators_1_to_2(::Val{:SpecialFunctions}) = ()

end
