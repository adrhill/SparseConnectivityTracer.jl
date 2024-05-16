module SparseConnectivityTracer

using ADTypes: ADTypes
import SparseArrays: sparse
import Random: rand, AbstractRNG, SamplerType

using DocStringExtensions

include("settypes/duplicatevector.jl")
include("settypes/recursiveset.jl")
include("settypes/sortedvector.jl")

include("tracers.jl")
include("conversion.jl")
include("operators.jl")
include("overload_connectivity.jl")
include("overload_gradient.jl")
include("overload_hessian.jl")
include("overload_dual.jl")
include("pattern.jl")
include("adtypes.jl")

export connectivity_pattern
export jacobian_pattern, local_jacobian_pattern
export hessian_pattern, local_hessian_pattern

# ADTypes interface
export TracerSparsityDetector

end # module
