module SparseConnectivityTracer
import Random: rand, AbstractRNG, SamplerType

struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

# We have to be careful when defining constructors:
# some code expecting numbers will convert Numbers `x` using `Tracer(x)`.
Tracer()          = Tracer(Set{UInt64}())
Tracer(::Number)  = Tracer()
Tracer(t::Tracer) = t

# we therefore use lower-case for internally convenience constructors
tracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))
tracer(index::Integer)       = Tracer(Set{UInt64}(index))

inputs(t::Tracer) = sort(Int.(keys(t.inputs.dict)))

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, inputs(t), "Tracer(", ',', ')', true)
end

include("conversion.jl")
include("operators.jl")
include("connectivity.jl")

export Tracer, inputs
export tracer, trace # TODO: don't export
export connectivity

end # module
