module SparseConnectivityTracerSpecialFunctionsExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using SpecialFunctions
else
    import ..SparseConnectivityTracer as SCT
    using ..SpecialFunctions
end

#=
Complex functions are ignored.
Functions with more than 2 arguments are ignored.
Functions with integer arguments are ignored.
=#

## 1-to-1

ops_1_to_1_s = (
    # Gamma Function
    gamma,
    loggamma,
    logabsgamma,
    logfactorial,
    digamma,
    invdigamma,
    trigamma,
    # Exponential and Trigonometric Integrals
    expinti,
    expintx,
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
    airy,
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
    # Zeta and Related Functions
    eta,
    zeta,
)

for op in ops_1_to_1_s
    T = typeof(op)
    @eval SCT.is_influence_zero_global(::$T) = false
    @eval SCT.is_firstder_zero_global(::$T) = false
    @eval SCT.is_seconder_zero_global(::$T) = false
end

ops_1_to_1 = ops_1_to_1_s

## 2-to-1

ops_2_to_1_ssc = (
    # Gamma Function
    gamma,
    loggamma,
    beta,
    logbeta,
    logabsbeta,
    # Exponential and Trigonometric Integrals
    expint,
    # Error functions, Dawson's and Fresnel Integrals
    erf,
    # Bessel Functions
    besselj,
    besseljx,
    sphericalbesselj,
    bessely,
    besselyx,
    sphericalbessely,
    hankelh1,
    hankelh1x,
    hankelh2,
    hankelh2x,
    besseli,
    besselix,
    besselk,
    besselkx,
)

for op in ops_2_to_1_ssc
    T = typeof(op)
    @eval SCT.is_influence_arg1_zero_global(::$T) = false
    @eval SCT.is_influence_arg2_zero_global(::$T) = false
    @eval SCT.is_firstder_arg1_zero_global(::$T) = false
    @eval SCT.is_seconder_arg1_zero_global(::$T) = false
    @eval SCT.is_firstder_arg2_zero_global(::$T) = false
    @eval SCT.is_seconder_arg2_zero_global(::$T) = false
    @eval SCT.is_crossder_zero_global(::$T) = false
end

ops_2_to_1 = ops_2_to_1_ssc

## Lists

SCT.list_operators_1_to_1(::Val{:SpecialFunctions}) = ops_1_to_1
SCT.list_operators_2_to_1(::Val{:SpecialFunctions}) = ops_2_to_1
SCT.list_operators_1_to_2(::Val{:SpecialFunctions}) = ()

function __init__()
    SCT.overload_all(SpecialFunctions)
    return nothing
end

end
