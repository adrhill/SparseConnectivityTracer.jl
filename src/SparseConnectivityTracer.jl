module SparseConnectivityTracer

using ADTypes: ADTypes, jacobian_sparsity, hessian_sparsity
using SparseArrays: SparseArrays
using SparseArrays: sparse
using Random: AbstractRNG, SamplerType

using LinearAlgebra: LinearAlgebra, Symmetric
using LinearAlgebra: Diagonal, diag, diagind
using FillArrays: Fill

using DocStringExtensions: DocStringExtensions, TYPEDEF, TYPEDFIELDS

include("settypes/recursiveset.jl")
include("settypes/sortedvector.jl")

include("patterns.jl")
include("tracers.jl")
include("exceptions.jl")
include("operators.jl")

include("overloads/conversion.jl")
include("overloads/gradient_tracer.jl")
include("overloads/hessian_tracer.jl")
include("overloads/utils.jl")
include("overloads/special_cases.jl")
include("overloads/three_arg.jl")
include("overloads/ifelse_global.jl")
include("overloads/dual.jl")
include("overloads/arrays.jl")
include("overloads/ambiguities.jl")

include("trace_functions.jl")
include("parse_outputs_to_matrix.jl")
include("adtypes_interface.jl")

export TracerSparsityDetector
export TracerLocalSparsityDetector
# Reexport ADTypes interface
export jacobian_sparsity, hessian_sparsity

export jacobian_eltype, hessian_eltype
export jacobian_buffer, hessian_buffer

end # module
