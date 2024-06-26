using SparseConnectivityTracer:
    AbstractTracer, ConnectivityTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: IndexSetVector, IndexSetHessian
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector

VECTOR_PATTERNS = (
    IndexSetVector{Int,BitSet},
    IndexSetVector{Int,Set{Int}},
    IndexSetVector{Int,DuplicateVector{Int}},
    IndexSetVector{Int,SortedVector{Int}},
)

HESSIAN_PATTERNS = (
    IndexSetHessian{Int,BitSet,Set{Tuple{Int,Int}}},
    IndexSetHessian{Int,Set{Int},Set{Tuple{Int,Int}}},
    IndexSetHessian{Int,DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}}},
    IndexSetHessian{Int,SortedVector{Int},SortedVector{Tuple{Int,Int}}},
    # TODO: test on RecursiveSet
)

CONNECTIVITY_TRACERS = (ConnectivityTracer{P} for P in VECTOR_PATTERNS)
GRADIENT_TRACERS = (GradientTracer{P} for P in VECTOR_PATTERNS)
HESSIAN_TRACERS = (HessianTracer{P} for P in HESSIAN_PATTERNS)
