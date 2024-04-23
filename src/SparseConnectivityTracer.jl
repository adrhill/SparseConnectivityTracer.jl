module SparseConnectivityTracer

using ADTypes: ADTypes
import Random: rand, AbstractRNG, SamplerType
import SparseArrays: sparse

abstract type AbstractTracer <: Number end

include("tracers.jl")
include("conversion.jl")
include("operators.jl")
include("overload_connectivity.jl")
include("overload_jacobian.jl")
include("pattern.jl")
include("adtypes.jl")

export JacobianTracer, ConnectivityTracer
export tracer, trace_input
export inputs
export pattern
export TracerSparsityDetector

end # module
