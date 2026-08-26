#===============#
# FillArrays.jl #
#===============#

# Since v1.17, FillArrays' reduction fast paths call `iszero`/`isone` on the fill value,
# which global tracers deliberately answer with a `MissingPrimalError`,
# so reductions of tracer `Fill`s get exact O(1) overloads.

## Sum
function Base.sum(A::AbstractFill{T}; dims = :) where {T <: AbstractTracer}
    t = isempty(A) ? myempty(T) : getindex_value(A)
    dims isa Colon && return t
    return Fill(t, Base.reduced_indices(axes(A), dims))
end

## Product
# `second_order_or` degrades to a plain union on `GradientTracer`s,
# so a single method is exact for both tracer types.
function Base.prod(A::AbstractFill{T}; dims = :) where {T <: AbstractTracer}
    t = getindex_value(A)
    if dims isa Colon
        isempty(A) && return myempty(T)
        length(A) == 1 && return t
        return second_order_or(t, t)
    end
    ri = Base.reduced_indices(axes(A), dims)
    isempty(A) && return Fill(myempty(T), ri)
    length(A) == prod(length, ri) && return Fill(t, ri) # one element per output slice
    return Fill(second_order_or(t, t), ri)
end
