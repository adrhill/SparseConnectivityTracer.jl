using SparseConnectivityTracer: AbstractTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: IndexSetGradientPattern, IndexSetHessianPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector

GRADIENT_PATTERNS = (
    IndexSetGradientPattern{Int,BitSet},
    IndexSetGradientPattern{Int,Set{Int}},
    IndexSetGradientPattern{Int,DuplicateVector{Int}},
    IndexSetGradientPattern{Int,SortedVector{Int}},
)

HESSIAN_PATTERNS = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},true},
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},false},
    IndexSetHessianPattern{Int,Set{Int},Set{Tuple{Int,Int}},false},
    IndexSetHessianPattern{Int,DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}},false},
    IndexSetHessianPattern{Int,SortedVector{Int},SortedVector{Tuple{Int,Int}},false},
    # TODO: test on RecursiveSet
)

GRADIENT_TRACERS = (GradientTracer{P} for P in GRADIENT_PATTERNS)
HESSIAN_TRACERS = (HessianTracer{P} for P in HESSIAN_PATTERNS)
