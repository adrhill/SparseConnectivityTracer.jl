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
        return P(union(gradient(px), gradient(py))) # return pattern
    end

    function output_union(px::P, py::P) where {P<:AbstractHessianPattern}
        return output_union(px, py, shared(P))
    end

    function output_union(px::P, py::P, ::Shared) where {P<:AbstractHessianPattern}
        g_out = union(gradient(px), gradient(py))
        hx, hy = hessian(px), hessian(py)
        hx !== hy && error("Expected shared Hessians, got $hx, $hy.")
        return P(g_out, hx) # return pattern
    end

    function output_union(px::P, py::P, ::NotShared) where {P<:IndexSetHessianPattern}
        g_out = union(gradient(px), gradient(py))
        h_out = union(hessian(px), hessian(py))
        return P(g_out, h_out) # return pattern
    end
    function output_union(px::P, py::P, ::NotShared) where {P<:DictHessianPattern}
        g_out = union(gradient(px), gradient(py))
        h_out = myunion!(deepcopy(hessian(px)), hessian(py))
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
