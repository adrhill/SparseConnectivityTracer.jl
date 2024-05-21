module SparseConnectivityTracer

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
include("pattern.jl")
include("adtypes.jl")

export connectivity_pattern, local_connectivity_pattern
export jacobian_pattern, local_jacobian_pattern
export hessian_pattern, local_hessian_pattern

# ADTypes interface
export TracerSparsityDetector

function overload_all(m::Module)
    for op in list_operators_1_to_1(Val(Symbol(m)))
        overload_connectivity_1_to_1(m, op)
        overload_gradient_1_to_1(m, op)
        overload_hessian_1_to_1(m, op)
    end
    for op in list_operators_2_to_1(Val(Symbol(m)))
        overload_connectivity_2_to_1(m, op)
        overload_gradient_2_to_1(m, op)
        overload_hessian_2_to_1(m, op)
    end
    for op in list_operators_1_to_2(Val(Symbol(m)))
        overload_connectivity_1_to_2(m, op)
        overload_gradient_1_to_2(m, op)
        overload_hessian_1_to_2(m, op)
    end
end

function __init__()
    overload_all(Base)

    @static if !isdefined(Base, :get_extension)
        @require SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b" include(
            "../ext/SparseConnectivityTracerSpecialFunctionsExt/SparseConnectivityTracerSpecialFunctionsExt.jl",
        )
    end
end

end # module
