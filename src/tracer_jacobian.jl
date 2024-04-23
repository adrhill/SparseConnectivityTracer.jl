"""
    JacobianTracer(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

See also the convenience constructor [`jacobiantracer`](@ref).
For a higher-level interface, refer to [`connectivity`](@ref).
"""
struct JacobianTracer <: AbstractTracer
    inputs::BitSet # indices of connected, enumerated inputs
end

const EMPTY_JACOBIAN_TRACER   = JacobianTracer(BitSet())
empty(::Type{JacobianTracer}) = EMPTY_CONNECTIVITY_TRACER
empty(::JacobianTracer)       = EMPTY_CONNECTIVITY_TRACER

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `JacobianTracer`.
# When this happens, we create a new empty tracer with no input connectivity.
JacobianTracer(::Number) = EMPTY_JACOBIAN_TRACER
JacobianTracer(t::JacobianTracer) = t

function uniontracer(a::JacobianTracer, b::JacobianTracer)
    return JacobianTracer(union(a.inputs, b.inputs))
end

"""
    jacobiantracer(index)
    jacobiantracer(indices)

Convenience constructor for [`JacobianTracer`](@ref) from input indices.
"""
jacobiantracer(index::Integer) = JacobianTracer(BitSet(index))
jacobiantracer(inds::NTuple{N,<:Integer}) where {N} = JacobianTracer(BitSet(inds))
jacobiantracer(inds...)                             = jacobiantracer(inds)

# Utilities for accessing input indices
inputs(t::JacobianTracer) = collect(t.inputs)

function Base.show(io::IO, t::JacobianTracer)
    return Base.show_delim_array(io, inputs(t), "JacobianTracer(", ',', ')', true)
end
