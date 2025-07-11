function Base.ifelse(::AbstractTracer, x, y)
    size(x) != size(y) && throw(
        DimensionMismatch(
            "Output sizes of x and y in `ifelse(condition, x, y)` don't match in size."
        ),
    )
    return output_union(x, y)
end

## output union on scalar outputs
function output_union(tx::T, ty::T) where {T <: GradientTracer}
    return T(union(gradient(tx), gradient(ty))) # return tracer
end

function output_union(tx::T, ty::T) where {T <: HessianTracer}
    return T(output_union(pattern(tx), pattern(ty), shared(T))) # return tracer
end
function output_union(tx::T, ty::T, ::Shared) where {T <: HessianTracer}
    g_out = union(gradient(tx), gradient(ty))
    hx, hy = hessian(tx), hessian(ty)
    hx !== hy && error("Expected shared Hessians, got $hx, $hy.")
    return T(g_out, hx) # return pattern
end

function output_union(tx::T, ty::T, ::NotShared) where {I <: Integer, G <: AbstractSet{I}, H <: AbstractSet{Tuple{I, I}}, T <: HessianTracer{I, G, H}}
    g_out = union(gradient(tx), gradient(ty))
    h_out = union(hessian(tx), hessian(ty))
    return T(g_out, h_out) # return pattern
end
function output_union(tx::T, ty::T, ::NotShared) where {I <: Integer, G <: AbstractSet{I}, H <: AbstractDict{I, G}, T <: HessianTracer{I, G, H}}
    g_out = union(gradient(tx), gradient(ty))
    h_out = myunion!(deepcopy(hessian(tx)), hessian(ty))
    return T(g_out, h_out) # return pattern
end

output_union(tx::AbstractTracer, y) = tx
output_union(x, ty::AbstractTracer) = ty

## output union on AbstractArray outputs
# TODO: add test
function output_union(tx::AbstractArray{T}, ty::AbstractArray{T}) where {T <: AbstractTracer}
    return output_union.(tx, ty)
end
function output_union(tx::AbstractArray{T}, y::AbstractArray) where {T <: AbstractTracer}
    return tx
end
function output_union(x::AbstractArray, ty::AbstractArray{T}) where {T <: AbstractTracer}
    return ty
end
