module SparseConnectivityTracer

using ADTypes: ADTypes
import Random: rand, AbstractRNG, SamplerType
import SparseArrays: sparse

abstract type AbstractTracer <: Number end

include("tracer_connectivity.jl")
include("tracer_jacobian.jl")
include("conversion.jl")
include("operators.jl")
include("overload_connectivity.jl")
include("overload_jacobian.jl")
include("connectivity.jl")
include("adtypes.jl")

export ConnectivityTracer, connectivitytracer
export JacobianTracer, jacobiantracer
export trace_input
export inputs
export connectivity
export TracerSparsityDetector

end # module
