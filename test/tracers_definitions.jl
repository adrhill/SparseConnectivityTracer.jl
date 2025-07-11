using SparseConnectivityTracer: DEFAULT_GRADIENT_TRACER, DEFAULT_HESSIAN_TRACER
using SparseConnectivityTracer: AbstractTracer, GradientTracer, HessianTracer, Dual
using SparseConnectivityTracer: RecursiveSet, SortedVector
using SparseConnectivityTracer: Shared, NotShared

GRADIENT_TRACERS = (
    GradientTracer{Int, BitSet},
    GradientTracer{Int, Set{Int}},
    GradientTracer{Int, SortedVector{Int}},
    GradientTracer{Int, RecursiveSet{Int}},
)

HESSIAN_TRACERS_SHARED = (
    HessianTracer{Int, BitSet, Set{Tuple{Int, Int}}, Shared},
    HessianTracer{Int, BitSet, Dict{Int, BitSet}, Shared},
)
HESSIAN_TRACERS_NOTSHARED = (
    HessianTracer{Int, BitSet, Set{Tuple{Int, Int}}, NotShared},
    HessianTracer{Int, SortedVector{Int}, SortedVector{Tuple{Int, Int}}, NotShared},
    HessianTracer{Int, BitSet, Dict{Int, BitSet}, NotShared},
    HessianTracer{Int, Set{Int}, Dict{Int, Set{Int}}, NotShared},
)
HESSIAN_TRACERS = union(HESSIAN_TRACERS_SHARED, HESSIAN_TRACERS_NOTSHARED)
