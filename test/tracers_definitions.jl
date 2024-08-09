using SparseConnectivityTracer: AbstractTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: IndexSetGradientPattern, IndexSetHessianPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using SparseConnectivityTracer: Shared, NotShared

GRADIENT_PATTERNS = (
    IndexSetGradientPattern{Int,BitSet},
    IndexSetGradientPattern{Int,Set{Int}},
    IndexSetGradientPattern{Int,DuplicateVector{Int}},
    IndexSetGradientPattern{Int,SortedVector{Int}},
)

HESSIAN_PATTERNS_SHARED = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},Shared},
)
HESSIAN_PATTERNS_NOTSHARED = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},NotShared},
    IndexSetHessianPattern{Int,Set{Int},Set{Tuple{Int,Int}},NotShared},
    IndexSetHessianPattern{
        Int,DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}},NotShared
    },
    IndexSetHessianPattern{Int,SortedVector{Int},SortedVector{Tuple{Int,Int}},NotShared},
    # TODO: test on RecursiveSet
)
HESSIAN_PATTERNS = union(HESSIAN_PATTERNS_SHARED, HESSIAN_PATTERNS_NOTSHARED)

GRADIENT_TRACERS = (GradientTracer{P} for P in GRADIENT_PATTERNS)
HESSIAN_TRACERS = (HessianTracer{P} for P in HESSIAN_PATTERNS)
