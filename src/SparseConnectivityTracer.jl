module SparseConnectivityTracer
import Random: rand, AbstractRNG, SamplerType

struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

Tracer() = Tracer(Set{UInt64}())
Tracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

tracer(index::Integer) = Tracer(Set{UInt64}(index)) # lower-case convenience constructor
inputs(t::Tracer) = sort(Int.(keys(t.inputs.dict)))

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, inputs(t), "Tracer(", ',', ')', true)
end

include("operators.jl")
include("connectivity.jl")

export Tracer, trace, inputs
export connectivity

end # module
