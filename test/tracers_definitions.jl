using SparseConnectivityTracer:
    AbstractTracer, ConnectivityTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: IndexSetGradientPattern, IndexSetHessianPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector

VECTOR_PATTERNS = (
    IndexSetGradientPattern{Int,BitSet},
    IndexSetGradientPattern{Int,Set{Int}},
    IndexSetGradientPattern{Int,DuplicateVector{Int}},
    IndexSetGradientPattern{Int,SortedVector{Int}},
)

HESSIAN_PATTERNS = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}}},
    IndexSetHessianPattern{Int,Set{Int},Set{Tuple{Int,Int}}},
    IndexSetHessianPattern{Int,DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}}},
    IndexSetHessianPattern{Int,SortedVector{Int},SortedVector{Tuple{Int,Int}}},
    # TODO: test on RecursiveSet
)

CONNECTIVITY_TRACERS = (ConnectivityTracer{P} for P in VECTOR_PATTERNS)
GRADIENT_TRACERS = (GradientTracer{P} for P in VECTOR_PATTERNS)
HESSIAN_TRACERS = (HessianTracer{P} for P in HESSIAN_PATTERNS)
