"""
    ConnectivityTracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`connectivity`](@ref).
"""
struct ConnectivityTracer <: AbstractTracer
    inputs::BitSet # indices of connected, enumerated inputs
end

const EMPTY_CONNECTIVITY_TRACER   = ConnectivityTracer(BitSet())
empty(::Type{ConnectivityTracer}) = EMPTY_CONNECTIVITY_TRACER
empty(::ConnectivityTracer)       = EMPTY_CONNECTIVITY_TRACER

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input connectivity.
ConnectivityTracer(::Number) = EMPTY_CONNECTIVITY_TRACER
ConnectivityTracer(t::ConnectivityTracer) = t

function uniontracer(a::ConnectivityTracer, b::ConnectivityTracer)
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

"""
    connectivitytracer(index)
    connectivitytracer(indices)

Convenience constructor for [`ConnectivityTracer`](@ref) from input indices.
"""
connectivitytracer(index::Integer) = ConnectivityTracer(BitSet(index))
connectivitytracer(inds::NTuple{N,<:Integer}) where {N} = ConnectivityTracer(BitSet(inds))
connectivitytracer(inds...)                             = connectivitytracer(inds)

# Utilities for accessing input indices
"""
    inputs(tracer)

Return raw `UInt64` input indices of a [`ConnectivityTracer`](@ref) or [`JacobianTracer`](@ref)

## Example
```jldoctest
julia> t = connectivitytracer(1, 2, 4)
ConnectivityTracer(1, 2, 4)

julia> inputs(t)
3-element Vector{Int64}:
 1
 2
 4
```
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(io, inputs(t), "ConnectivityTracer(", ',', ')', true)
end
