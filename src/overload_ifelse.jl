const SimpleTracer = Union{ConnectivityTracer,GradientTracer,HessianTracer}

function Base.ifelse(::T, x, y) where {T<:SimpleTracer}
    size(x) != size(y) &&
        throw(DimensionMismatch("Outputs sizes of `ifelse` arguments don't match in size."))
    return output_union(x, y)
end

## output union on scalar outputs
output_union(tx::C, ty::C) where {C<:ConnectivityTracer} = C(inputs(tx) ∪ inputs(ty))
output_union(tx::G, ty::G) where {G<:GradientTracer} = G(gradient(tx) ∪ gradient(ty))
function output_union(tx::H, ty::H) where {H<:HessianTracer}
    return H(gradient(tx) ∪ gradient(ty), hessian(tx) ∪ hessian(ty))
end

output_union(tx::T, y) where {T<:SimpleTracer} = tx
output_union(x, ty::T) where {T<:SimpleTracer} = ty

## output union on AbstractArray outputs
function output_union(tx::AbstractArray{T}, ty::AbstractArray{T}) where {T<:SimpleTracer}
    return output_union.(tx, ty)
end
function output_union(tx::AbstractArray{T}, y::AbstractArray) where {T<:SimpleTracer}
    return tx
end
function output_union(x::AbstractArray, ty::AbstractArray{T}) where {T<:SimpleTracer}
    return ty
end
