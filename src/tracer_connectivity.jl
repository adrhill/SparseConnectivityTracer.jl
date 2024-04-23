"""
    Tracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`connectivity`](@ref).
"""
struct Tracer <: Number
    inputs::BitSet # indices of connected, enumerated inputs
end

const EMPTY_TRACER = Tracer(BitSet())

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `Tracer`.
# When this happens, we create a new empty tracer with no input connectivity.
Tracer(::Number)  = EMPTY_TRACER
Tracer(t::Tracer) = t

uniontracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

"""
    tracer(index)
    tracer(indices)

Convenience constructor for [`Tracer`](@ref) from input indices.
"""
tracer(index::Integer) = Tracer(BitSet(index))
tracer(inds::NTuple{N,<:Integer}) where {N} = Tracer(BitSet(inds))
tracer(inds...)                             = tracer(inds)

# Utilities for accessing input indices
"""
    inputs(tracer)

Return raw `UInt64` input indices of a [`Tracer`](@ref) or [`JacobianTracer`](@ref)

## Example
```jldoctest
julia> t = tracer(1, 2, 4)
Tracer(1, 2, 4)

julia> inputs(t)
3-element Vector{Int64}:
 1
 2
 4
```
"""
inputs(t::Tracer) = collect(t.inputs)

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, inputs(t), "Tracer(", ',', ')', true)
end
