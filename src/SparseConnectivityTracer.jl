module SparseConnectivityTracer

using ADTypes: ADTypes
import SparseArrays: sparse
import Random: rand, AbstractRNG, SamplerType

include("tracers.jl")
include("conversion.jl")
include("operators.jl")
include("overload_connectivity.jl")
include("overload_jacobian.jl")
include("overload_hessian.jl")
include("pattern.jl")
include("adtypes.jl")

include("settypes/duplicatevector.jl")
include("settypes/recursiveset.jl")
include("settypes/sortedvector.jl")

export ConnectivityTracer, connectivity_pattern
export JacobianTracer, jacobian_pattern
export HessianTracer, hessian_pattern

export TracerSparsityDetector

end # module
