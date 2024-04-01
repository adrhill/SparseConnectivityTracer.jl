module SparseConnectivityTracer
import Random: rand, AbstractRNG, SamplerType
import SparseArrays: sparse

struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `Tracer`.
# When this happens, we create a new empty tracer with no input connectivity.
Tracer(::Number)  = tracer()
Tracer(t::Tracer) = t

# We therefore exclusively use lower-case for internal convenience constructors
"""
    trace(index)
    trace(indices)

Convenience constructor for [`Trace`](@ref) from input indices.
"""
tracer()                     = Tracer(Set{UInt64}())
tracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

tracer(index::Integer)                      = Tracer(Set{UInt64}(index))
tracer(inds::NTuple{N,<:Integer}) where {N} = Tracer(Set{UInt64}(inds))
tracer(inds...)                             = tracer(inds)

# Utilities for accessing input indices
inputs(t::Tracer) = collect(keys(t.inputs.dict))
sortedinputs(t::Tracer) = sortedinputs(Int, t)
sortedinputs(::Type{T}, t::Tracer) where {T<:Integer} = T.(sort!(inputs(t)))

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, sortedinputs(Int, t), "Tracer(", ',', ')', true)
end

include("conversion.jl")
include("operators.jl")
include("connectivity.jl")

export Tracer, tracer, inputs, sortedinputs
export connectivity

end # module
