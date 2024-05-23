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

## Overload comparisons
# We don't return empty tracers since the resulting Bool could be used as a Number. 
for fn in (:isequal, :isapprox, :isless, :(==), :(<), :(>), :(<=), :(>=))
    @eval Base.$fn(tx::C, ty::C) where {C<:ConnectivityTracer} = C(inputs(tx) ∪ inputs(ty))
    @eval Base.$fn(tx::G, ty::G) where {G<:GradientTracer} = G(gradient(tx) ∪ gradient(ty))
    @eval function Base.$fn(tx::T, ty::T) where {G,H,T<:HessianTracer{G,H}}
        return T(gradient(tx) ∪ gradient(ty), empty(H))
    end

    @eval Base.$fn(tx::T, y::Real) where {T<:SimpleTracer} = tx
    @eval Base.$fn(x::Real, ty::T) where {T<:SimpleTracer} = ty
end
