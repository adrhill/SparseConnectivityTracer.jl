# NNlib activation functions on tracers.
# Parametric activation functions with two or more arguments are ignored.
module SparseConnectivityTracerNNlibExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using NNlib:
        NNlib,
        celu,
        elu,
        gelu,
        hardswish,
        hardtanh,
        hardσ,
        leakyrelu,
        lisht,
        logcosh,
        logσ,
        mish,
        relu,
        relu6,
        selu,
        sigmoid_fast,
        softplus,
        softshrink,
        softsign,
        swish,
        tanh_fast,
        tanhshrink,
        trelu,
        σ
else
    import ..SparseConnectivityTracer as SCT
    using ..NNlib:
        NNlib,
        celu,
        elu,
        gelu,
        hardswish,
        hardtanh,
        hardσ,
        leakyrelu,
        lisht,
        logcosh,
        logσ,
        mish,
        relu,
        relu6,
        selu,
        sigmoid_fast,
        softplus,
        softshrink,
        softsign,
        swish,
        tanh_fast,
        tanhshrink,
        trelu,
        σ
end

## 1-to-1

# ops_1_to_1_s: 
# x -> f  != 0
# ∂f/∂x   != 0
# ∂²f/∂x² != 0
ops_1_to_1_s = (
    # ReLU-like activation functions
    celu,
    elu,
    gelu,
    selu,
    # Other activation functions
    σ, # sigmoid
    hardswish,
    lisht,
    logσ,
    logcosh,
    mish,
    sigmoid_fast,
    softplus,
    softsign,
    swish,
    tanh_fast,
    tanhshrink,
)

for op in ops_1_to_1_s
    T = typeof(op)
    @eval SCT.is_der1_zero_global(::$T) = false
    @eval SCT.is_der2_zero_global(::$T) = false
end

SCT.is_der2_zero_local(::typeof(celu), x) = x > 0
SCT.is_der2_zero_local(::typeof(elu), x) = x > 0
SCT.is_der2_zero_local(::typeof(selu), x) = x > 0

SCT.is_der1_zero_local(::typeof(hardswish), x) = x < -3
SCT.is_der2_zero_local(::typeof(hardswish), x) = x < -3 || x > 3

# ops_1_to_1_f:
# x -> f  != 0
# ∂f/∂x   != 0
# ∂²f/∂x² == 0
ops_1_to_1_f = (
    # ReLU-like activation functions
    leakyrelu,
    relu,
    relu6,
    trelu,
    # Other activation functions
    hardσ,
    hardtanh,
    softshrink,
)

for op in ops_1_to_1_f
    T = typeof(op)
    @eval SCT.is_der1_zero_global(::$T) = false
    @eval SCT.is_der2_zero_global(::$T) = true
end

SCT.is_der1_zero_local(::typeof(relu), x) = x < 0
SCT.is_der1_zero_local(::typeof(relu6), x) = x < 0 || x > 6
SCT.is_der1_zero_local(::typeof(trelu), x) = x < 1

SCT.is_der1_zero_local(::typeof(hardσ), x) = x < -3 || x > 3
SCT.is_der1_zero_local(::typeof(hardtanh), x) = x < -1 || x > 1
SCT.is_der1_zero_local(::typeof(softshrink), x) = x > -0.5 && x < 0.5

ops_1_to_1 = union(ops_1_to_1_s, ops_1_to_1_f)

## Overload
eval(SCT.generate_code_1_to_1(:NNlib, ops_1_to_1))

## List operators for later testing
SCT.test_operators_1_to_1(::Val{:NNlib}) = ops_1_to_1
SCT.test_operators_2_to_1(::Val{:NNlib}) = ()
SCT.test_operators_1_to_2(::Val{:NNlib}) = ()

end
