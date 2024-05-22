# NNlib activation functions on tracers.
# Parametric activation functions with two or more arguments are ignored.
module SparseConnectivityTracerNNlibExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using NNlib
else
    import ..SparseConnectivityTracer as SCT
    using ..NNlib
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
    @eval SCT.is_influence_zero_global(::$T) = false
    @eval SCT.is_firstder_zero_global(::$T) = false
    @eval SCT.is_seconder_zero_global(::$T) = false
end

SCT.is_seconder_zero_local(::typeof(celu)) = x > 0
SCT.is_seconder_zero_local(::typeof(elu)) = x > 0
SCT.is_seconder_zero_local(::typeof(selu)) = x > 0

SCT.is_firstder_zero_local(::typeof(hardswish)) = x < -3
SCT.is_seconder_zero_local(::typeof(hardswish)) = x < -3 || x > 3

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
    @eval SCT.is_influence_zero_global(::$T) = false
    @eval SCT.is_firstder_zero_global(::$T) = false
    @eval SCT.is_seconder_zero_global(::$T) = true
end

SCT.is_firstder_zero_local(::typeof(relu), x) = x < 0
SCT.is_firstder_zero_local(::typeof(relu6), x) = x < 0 || x > 6
SCT.is_firstder_zero_local(::typeof(trelu), x) = x < 1

SCT.is_firstder_zero_local(::typeof(hardσ), x) = x < -3 || x > 3
SCT.is_firstder_zero_local(::typeof(hardtanh), x) = x < -1 || x > 1
SCT.is_firstder_zero_local(::typeof(softshrink), x) = x > -0.5 && x < 0.5

ops_1_to_1 = union(ops_1_to_1_s, ops_1_to_1_f)

## Lists

SCT.list_operators_1_to_1(::Val{:NNlib}) = ops_1_to_1
SCT.list_operators_2_to_1(::Val{:NNlib}) = ()
SCT.list_operators_1_to_2(::Val{:NNlib}) = ()

## Overloads

eval(SCT.overload_all(:NNlib))

end
