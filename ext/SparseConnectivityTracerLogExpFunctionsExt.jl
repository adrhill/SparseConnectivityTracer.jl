module SparseConnectivityTracerLogExpFunctionsExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using LogExpFunctions
else
    import ..SparseConnectivityTracer as SCT
    using ..LogExpFunctions
end

# 1-to-1 functions

ops_1_to_1 = (
    xlogx,
    xexpx,
    logistic,
    logit,
    logcosh,
    logabssinh,
    log1psq,
    log1pexp,
    log1mexp,
    log2mexp,
    logexpm1,
    softplus,
    invsoftplus,
    log1pmx,
    logmxp1,
    cloglog,
    cexpexp,
    loglogistic,
    logitexp,
    log1mlogistic,
    logit1mexp,
)

for op in ops_1_to_1
    T = typeof(op)
    @eval SCT.is_der1_zero_global(::$T) = false
    @eval SCT.is_der2_zero_global(::$T) = false
end

# 2-to-1 functions

ops_2_to_1 = (xlogy, xlog1py, xexpy, logaddexp, logsubexp)

for op in ops_2_to_1
    T = typeof(op)
    @eval SCT.is_der1_arg1_zero_global(::$T) = false
    @eval SCT.is_der2_arg1_zero_global(::$T) = false
    @eval SCT.is_der1_arg2_zero_global(::$T) = false
    @eval SCT.is_der2_arg2_zero_global(::$T) = false
    @eval SCT.is_der_cross_zero_global(::$T) = false
end

# Generate overloads
eval(SCT.generate_code_1_to_1(:LogExpFunctions, ops_1_to_1))
eval(SCT.generate_code_2_to_1(:LogExpFunctions, ops_2_to_1))

# List operators for later testing
SCT.test_operators_1_to_1(::Val{:LogExpFunctions}) = ops_1_to_1
SCT.test_operators_2_to_1(::Val{:LogExpFunctions}) = ops_2_to_1
SCT.test_operators_1_to_2(::Val{:LogExpFunctions}) = ()

end
