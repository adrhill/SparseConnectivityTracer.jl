#==============#
# Connectivity #
#==============#

"""
    ConnectivityTracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`pattern`](@ref).
"""
struct ConnectivityTracer <: AbstractTracer
    inputs::BitSet # indices of connected, enumerated inputs
end

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(io, inputs(t), "ConnectivityTracer(", ',', ')', true)
end

const EMPTY_CONNECTIVITY_TRACER = ConnectivityTracer(BitSet())

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer(::Number) = EMPTY_CONNECTIVITY_TRACER
ConnectivityTracer(t::ConnectivityTracer) = t

#==========#
# Jacobian #
#==========#

"""
    JacobianTracer(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`pattern`](@ref).
"""
struct JacobianTracer <: AbstractTracer
    inputs::BitSet # indices of connected, enumerated inputs
end

function Base.show(io::IO, t::JacobianTracer)
    return Base.show_delim_array(io, inputs(t), "JacobianTracer(", ',', ')', true)
end

const EMPTY_JACOBIAN_TRACER = JacobianTracer(BitSet())

JacobianTracer(::Number) = EMPTY_JACOBIAN_TRACER
JacobianTracer(t::JacobianTracer) = t

#===========#
# Utilities #
#===========#

## Access inputs
"""
    inputs(tracer)

Return raw `UInt64` input indices of a [`ConnectivityTracer`](@ref) or [`JacobianTracer`](@ref)

## Example
```jldoctest
julia> t = tracer(ConnectivityTracer, 1, 2, 4)
ConnectivityTracer(1, 2, 4)

julia> inputs(t)
3-element Vector{Int64}:
 1
 2
 4
```
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::JacobianTracer) = collect(t.inputs)

## Unions of tracers
function uniontracer(a::ConnectivityTracer, b::ConnectivityTracer)
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

function uniontracer(a::JacobianTracer, b::JacobianTracer)
    return JacobianTracer(union(a.inputs, b.inputs))
end

## Get empty tracer
empty(::JacobianTracer)           = EMPTY_JACOBIAN_TRACER
empty(::Type{JacobianTracer})     = EMPTY_JACOBIAN_TRACER
empty(::ConnectivityTracer)       = EMPTY_CONNECTIVITY_TRACER
empty(::Type{ConnectivityTracer}) = EMPTY_CONNECTIVITY_TRACER

"""
    tracer(JacobianTracer, index)
    tracer(JacobianTracer, indices)
    tracer(ConnectivityTracer, index)
    tracer(ConnectivityTracer, indices)

Convenience constructor for [`JacobianTracer`](@ref) [`ConnectivityTracer`](@ref) from input indices.
"""
tracer(::Type{JacobianTracer}, index::Integer) = JacobianTracer(BitSet(index))
tracer(::Type{ConnectivityTracer}, index::Integer) = ConnectivityTracer(BitSet(index))

function tracer(::Type{JacobianTracer}, inds::NTuple{N,<:Integer}) where {N}
    return JacobianTracer(BitSet(inds))
end
function tracer(::Type{ConnectivityTracer}, inds::NTuple{N,<:Integer}) where {N}
    return ConnectivityTracer(BitSet(inds))
end

tracer(::Type{T}, inds...) where {T<:AbstractTracer} = tracer(T, inds)
