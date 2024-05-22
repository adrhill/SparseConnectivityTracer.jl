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
    # Activation functions
    σ, # sigmoid
    logσ,
    swish,
    hardswish,
    lisht,
    softsign,
    softplus,
    logcosh,
    mish,
    tanhshrink,
    # ReLU-like activation functions
    elu,
    gelu,
    selu,
    celu,
)

for op in ops_1_to_1_s
    T = typeof(op)
    @eval SCT.is_influence_zero_global(::$T) = false
    @eval SCT.is_firstder_zero_global(::$T) = false
    @eval SCT.is_seconder_zero_global(::$T) = false
end
is_seconder_zero_local(::typeof(elu)) = x > 0
is_seconder_zero_local(::typeof(selu)) = x > 0
is_seconder_zero_local(::typeof(celu)) = x > 0

is_firstder_zero_local(::typeof(hardswish)) = x < -3
is_seconder_zero_local(::typeof(hardswish)) = x < -3 || x > 3

# ops_1_to_1_f:
# x -> f  != 0
# ∂f/∂x   != 0
# ∂²f/∂x² == 0
ops_1_to_1_f = (
    # ReLU-like activation functions
    relu,
    relu6,
    trelu,
    leakyrelu,
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

is_firstder_zero_local(::typeof(relu), x) = x < 0
is_firstder_zero_local(::typeof(relu6), x) = x < 0 || x > 6
is_firstder_zero_local(::typeof(trelu), x) = x < 1

is_firstder_zero_local(::typeof(hardσ), x) = x < -3 || x > 3
is_firstder_zero_local(::typeof(hardtanh), x) = x < -1 || x > 1
is_firstder_zero_local(::typeof(softshrink), x) = x > -0.5 && x < 0.5

ops_1_to_1 = union(ops_1_to_1_s, ops_1_to_1_f)

## Lists

SCT.list_operators_1_to_1(::Val{:NNlib}) = ops_1_to_1
SCT.list_operators_2_to_1(::Val{:NNlib}) = ()
SCT.list_operators_1_to_2(::Val{:NNlib}) = ()

## Overloads

eval(SCT.overload_all(:NNlib))

end
