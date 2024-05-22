module SparseConnectivityTracer

const SCT = SparseConnectivityTracer

using ADTypes: ADTypes
using Compat: Returns
import SparseArrays: sparse
import Random: rand, AbstractRNG, SamplerType

using DocStringExtensions

if !isdefined(Base, :get_extension)
    using Requires
end

include("settypes/duplicatevector.jl")
include("settypes/recursiveset.jl")
include("settypes/sortedvector.jl")

include("tracers.jl")
include("exceptions.jl")
include("conversion.jl")
include("operators.jl")

include("overload_connectivity.jl")
include("overload_gradient.jl")
include("overload_hessian.jl")
include("overload_dual.jl")
include("overload_all.jl")

include("pattern.jl")
include("adtypes.jl")

export connectivity_pattern, local_connectivity_pattern
export jacobian_pattern, local_jacobian_pattern
export hessian_pattern, local_hessian_pattern

# ADTypes interface
export TracerSparsityDetector
export TracerLocalSparsityDetector

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b" include(
            "../ext/SparseConnectivityTracerSpecialFunctionsExt/SparseConnectivityTracerSpecialFunctionsExt.jl",
        )
    end
end

end # module
