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
    function output_union(tx::C, ty::C) where {C<:ConnectivityTracer}
        return C(union(inputs(tx), inputs(ty)))
    end
    function output_union(tx::G, ty::G) where {G<:GradientTracer}
        return G(union(gradient(tx), gradient(ty)))
    end
    function output_union(tx::H, ty::H) where {H<:HessianTracer}
        return H(union(gradient(tx), gradient(ty)), union(hessian(tx), hessian(ty)))
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
    @eval is_influence_arg1_zero_global(::$T) = false
    @eval is_influence_arg2_zero_global(::$T) = false
    @eval is_firstder_arg1_zero_global(::$T) = true
    @eval is_seconder_arg1_zero_global(::$T) = true
    @eval is_firstder_arg2_zero_global(::$T) = true
    @eval is_seconder_arg2_zero_global(::$T) = true
    @eval is_crossder_zero_global(::$T) = true

    op_symb = nameof(op)
    SparseConnectivityTracer.eval(overload_connectivity_2_to_1(:Base, op_symb))
    SparseConnectivityTracer.eval(overload_gradient_2_to_1(:Base, op_symb))
    SparseConnectivityTracer.eval(overload_hessian_2_to_1(:Base, op_symb))
end
