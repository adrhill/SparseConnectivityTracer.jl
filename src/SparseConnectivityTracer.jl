module SparseConnectivityTracer

using ADTypes: ADTypes, jacobian_sparsity, hessian_sparsity
using SparseArrays: SparseArrays
using SparseArrays: sparse
using Random: AbstractRNG, SamplerType

using LinearAlgebra: LinearAlgebra, Symmetric
using LinearAlgebra: Diagonal, diag, diagind
using FillArrays: Fill

using DocStringExtensions: DocStringExtensions, TYPEDEF, TYPEDFIELDS

if !isdefined(Base, :get_extension)
    using Requires
end

include("settypes/duplicatevector.jl")
include("settypes/recursiveset.jl")
include("settypes/sortedvector.jl")

include("patterns.jl")
include("tracers.jl")
include("exceptions.jl")
include("operators.jl")

include("overloads/conversion.jl")
include("overloads/gradient_tracer.jl")
include("overloads/hessian_tracer.jl")
include("overloads/ambiguities.jl")
include("overloads/special_cases.jl")
include("overloads/ifelse_global.jl")
include("overloads/dual.jl")
include("overloads/arrays.jl")
include("overloads/utils.jl")

include("trace_functions.jl")
include("adtypes_interface.jl")

export TracerSparsityDetector
export TracerLocalSparsityDetector
# Reexport ADTypes interface
export jacobian_sparsity, hessian_sparsity

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b" include(
            "../ext/SparseConnectivityTracerSpecialFunctionsExt.jl"
        )
        @require NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd" include(
            "../ext/SparseConnectivityTracerNNlibExt.jl"
        )
        @require LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688" include(
            "../ext/SparseConnectivityTracerLogExpFunctionsExt.jl"
        )
        @require NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3" include(
            "../ext/SparseConnectivityTracerNaNMathExt.jl"
        )
        # NOTE: SparseConnectivityTracerDataInterpolationsExt is not loaded on Julia <1.10
    end
end

end # module
