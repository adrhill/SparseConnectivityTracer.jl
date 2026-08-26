#===============#
# FillArrays.jl #
#===============#

# Since v1.17, FillArrays' reduction fast paths call `iszero`/`isone` on the fill value,
# which global tracers deliberately answer with a `MissingPrimalError`.
# `Fill`s of tracers are also SCT's own representation of "all outputs share one pattern"
# (see `inv`, `eigen`, `cholesky` and `\` in `arrays.jl`),
# so reductions get exact O(1) overloads that stay in this representation:
# a scalar tracer for a full reduction, a `Fill` for a reduction along `dims`.

## Sum
function Base.sum(A::AbstractFill{T}; dims = :) where {T <: AbstractTracer}
    # Summation is linear: no interactions between elements,
    # and the union of identical patterns is the pattern of the fill value itself.
    t = isempty(A) ? myempty(T) : getindex_value(A)
    dims isa Colon && return t
    return Fill(t, Base.reduced_indices(axes(A), dims))
end

## Product
# Only three cases matter for the product over a (sub-)`Fill`:
# an empty product is `one(T)` and has no dependencies,
# a single-element product is the fill value itself,
# and everything else contains all self-interactions after one `second_order_or`
# (which on `GradientTracer`s degrades to the plain union, so this is exact for both tracer types).
function Base.prod(A::AbstractFill{T}; dims = :) where {T <: AbstractTracer}
    t = getindex_value(A)
    if dims isa Colon
        isempty(A) && return myempty(T)
        length(A) == 1 && return t
        return second_order_or(t, t)
    end
    ri = Base.reduced_indices(axes(A), dims)
    isempty(A) && return Fill(myempty(T), ri) # a slice (or the output itself) is empty
    length(A) == prod(length, ri) && return Fill(t, ri) # one element per output slice
    return Fill(second_order_or(t, t), ri)
end
