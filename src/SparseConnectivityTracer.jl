module SparseConnectivityTracer

using ADTypes: ADTypes
import SparseArrays: sparse
import Random: rand, AbstractRNG, SamplerType

abstract type AbstractTracer <: Number end

include("tracers.jl")
include("conversion.jl")
include("operators.jl")
include("overload_connectivity.jl")
include("overload_jacobian.jl")
include("overload_hessian.jl")
include("pattern.jl")
include("adtypes.jl")

export JacobianTracer, ConnectivityTracer, HessianTracer
export tracer, trace_input
export inputs
export pattern
export TracerSparsityDetector

end # module
