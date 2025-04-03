module SparseConnectivityTracerNaNMathExt

import SparseConnectivityTracer as SCT
using NaNMath: NaNMath

## 1-to-1

# ops_1_to_1_s:
# x -> f  != 0
# ∂f/∂x   != 0
# ∂²f/∂x² != 0
ops_1_to_1_s = (
    NaNMath.sqrt,
    NaNMath.sin,
    NaNMath.cos,
    NaNMath.tan,
    NaNMath.asin,
    NaNMath.acos,
    NaNMath.acosh,
    NaNMath.atanh,
    NaNMath.log,
    NaNMath.log2,
    NaNMath.log10,
    NaNMath.log1p,
    NaNMath.lgamma,
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
ops_2_to_1_ssc = (NaNMath.pow,)

for op in ops_2_to_1_ssc
    T = typeof(op)
    @eval SCT.is_der1_arg1_zero_global(::$T) = false
    @eval SCT.is_der2_arg1_zero_global(::$T) = false
    @eval SCT.is_der1_arg2_zero_global(::$T) = false
    @eval SCT.is_der2_arg2_zero_global(::$T) = false
    @eval SCT.is_der_cross_zero_global(::$T) = false
end

# ops_2_to_1_ffz:
# ∂f/∂x    != 0
# ∂²f/∂x²  == 0
# ∂f/∂y    != 0
# ∂²f/∂y²  == 0
# ∂²f/∂x∂y == 0
ops_2_to_1_ffz = (NaNMath.max, NaNMath.min)

for op in ops_2_to_1_ffz
    T = typeof(op)
    @eval SCT.is_der1_arg1_zero_global(::$T) = false
    @eval SCT.is_der2_arg1_zero_global(::$T) = true
    @eval SCT.is_der1_arg2_zero_global(::$T) = false
    @eval SCT.is_der2_arg2_zero_global(::$T) = true
    @eval SCT.is_der_cross_zero_global(::$T) = true
end

SCT.is_der1_arg1_zero_local(::typeof(NaNMath.max), x, y) = x < y
SCT.is_der1_arg2_zero_local(::typeof(NaNMath.max), x, y) = y < x

SCT.is_der1_arg1_zero_local(::typeof(NaNMath.min), x, y) = x > y
SCT.is_der1_arg2_zero_local(::typeof(NaNMath.min), x, y) = y > x

ops_2_to_1 = union(ops_2_to_1_ssc, ops_2_to_1_ffz)

## Overloads
eval(SCT.generate_code_1_to_1(:NaNMath, ops_1_to_1))
eval(SCT.generate_code_2_to_1(:NaNMath, ops_2_to_1))

## List operators for later testing
SCT.test_operators_1_to_1(::Val{:NaNMath}) = ops_1_to_1
SCT.test_operators_2_to_1(::Val{:NaNMath}) = ops_2_to_1
SCT.test_operators_1_to_2(::Val{:NaNMath}) = ()

end # module
