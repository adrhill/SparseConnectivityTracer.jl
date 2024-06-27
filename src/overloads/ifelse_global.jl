@static if VERSION >= v"1.8"
    function Base.ifelse(::AbstractTracer, x, y)
        size(x) != size(y) && throw(
            DimensionMismatch(
                "Output sizes of x and y in `ifelse(condition, x, y)` don't match in size.",
            ),
        )
        return output_union(x, y)
    end

    ## output union on scalar outputs
    function output_union(tx::T, ty::T) where {T<:AbstractTracer}
        return T(output_union(pattern(tx), pattern(ty))) # return tracer
    end
    function output_union(px::P, py::P) where {P<:IndexSetGradientPattern}
        return P(union(set(px), set(py))) # return pattern
    end
    function output_union(px::P, py::P) where {P<:IndexSetHessianPattern}
        g_out = union(gradient(px), gradient(py))
        h_out = union(hessian(px), hessian(py))
        return P(g_out, h_out) # return pattern
    end

    output_union(tx::AbstractTracer, y) = tx
    output_union(x, ty::AbstractTracer) = ty

    ## output union on AbstractArray outputs
    function output_union(
        tx::AbstractArray{T}, ty::AbstractArray{T}
    ) where {T<:AbstractTracer}
        return output_union.(tx, ty)
    end
    function output_union(tx::AbstractArray{T}, y::AbstractArray) where {T<:AbstractTracer}
        return tx
    end
    function output_union(x::AbstractArray, ty::AbstractArray{T}) where {T<:AbstractTracer}
        return ty
    end
end

# Overload only on AbstractTracer, not Dual 
for op in (isequal, isapprox, isless, ==, <, >, <=, >=)
    T = typeof(op)
    @eval is_infl_arg1_zero_global(::$T) = false
    @eval is_infl_arg2_zero_global(::$T) = false
    @eval is_der1_arg1_zero_global(::$T) = true
    @eval is_der2_arg1_zero_global(::$T) = true
    @eval is_der1_arg2_zero_global(::$T) = true
    @eval is_der2_arg2_zero_global(::$T) = true
    @eval is_der_cross_zero_global(::$T) = true

    op_symb = nameof(op)
    SparseConnectivityTracer.eval(overload_connectivity_2_to_1(:Base, op_symb))
    SparseConnectivityTracer.eval(overload_gradient_2_to_1(:Base, op_symb))
    SparseConnectivityTracer.eval(overload_hessian_2_to_1(:Base, op_symb))
end
