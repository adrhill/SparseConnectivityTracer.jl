using SparseConnectivityTracer: AbstractTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: IndexSetGradientPattern
using SparseConnectivityTracer: IndexSetHessianPattern, DictHessianPattern
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using SparseConnectivityTracer: Shared, NotShared

GRADIENT_PATTERNS = (
    IndexSetGradientPattern{Int,BitSet},
    IndexSetGradientPattern{Int,Set{Int}},
    IndexSetGradientPattern{Int,DuplicateVector{Int}},
    IndexSetGradientPattern{Int,SortedVector{Int}},
    IndexSetGradientPattern{Int,RecursiveSet{Int}},
)

HESSIAN_PATTERNS_SHARED = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},Shared},
    DictHessianPattern{Int,BitSet,Dict{Int,BitSet},Shared},
)
HESSIAN_PATTERNS_NOTSHARED = (
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},NotShared},
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},NotShared},
    IndexSetHessianPattern{
        Int,DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}},NotShared
    },
    IndexSetHessianPattern{Int,SortedVector{Int},SortedVector{Tuple{Int,Int}},NotShared},
    # TODO: test on RecursiveSet
    DictHessianPattern{Int,BitSet,Dict{Int,BitSet},NotShared},
    DictHessianPattern{Int,Set{Int},Dict{Int,Set{Int}},NotShared},
)
HESSIAN_PATTERNS = union(HESSIAN_PATTERNS_SHARED, HESSIAN_PATTERNS_NOTSHARED)

GRADIENT_TRACERS = (GradientTracer{P} for P in GRADIENT_PATTERNS)
HESSIAN_TRACERS = (HessianTracer{P} for P in HESSIAN_PATTERNS)
