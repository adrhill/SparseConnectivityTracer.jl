module SparseConnectivityTracer
import Random: rand, AbstractRNG, SamplerType
import SparseArrays: sparse

include("tracer.jl")
include("conversion.jl")
include("operators.jl")
include("connectivity.jl")

export Tracer, tracer, inputs, sortedinputs
export connectivity

end # module
