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
# `second_order_or` on `GradientTracer`s degrades to the plain union,
# so a single method is exact for both tracer types.
function prod_of_fill(t::T, n::Integer) where {T <: AbstractTracer}
    n == 0 && return myempty(T) # empty product is `one(T)`: no dependencies
    n == 1 && return t
    return second_order_or(t, t) # all self-interactions appear after one application
end

function Base.prod(A::AbstractFill{T}; dims = :) where {T <: AbstractTracer}
    dims isa Colon && return prod_of_fill(getindex_value(A), length(A))
    # `reduced_indices` validates `dims` and collapses exactly the reduced axes to length 1,
    # so comparing against the original axes recovers the number of elements per output slice.
    ri = Base.reduced_indices(axes(A), dims)
    n = prod(map((ax, rax) -> ax == rax ? 1 : length(ax), axes(A), ri))
    return Fill(prod_of_fill(getindex_value(A), n), ri)
end
