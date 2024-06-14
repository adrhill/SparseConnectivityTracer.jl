using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, HessianTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet

SET_TYPES = (BitSet, Set{Int}, DuplicateVector{Int}, RecursiveSet{Int}, SortedVector{Int})

include("jacobian.jl")
include("hessian.jl")

SUITE = BenchmarkGroup()
for S1 in SET_TYPES
    S2 = Set{Tuple{Int,Int}}

    G = GradientTracer{S1}
    H = HessianTracer{S1,S2}

    SUITE["Jacobian"]["Global"][nameof(S1)] = jacbench(TracerSparsityDetector(G, H))
    SUITE["Jacobian"]["Local"][nameof(S1)] = jacbench(TracerLocalSparsityDetector(G, H))
    SUITE["Hessian"]["Global"][(nameof(S1), nameof(S2))] = hessbench(
        TracerSparsityDetector(G, H)
    )
    SUITE["Hessian"]["Local"][(nameof(S1), nameof(S2))] = hessbench(
        TracerLocalSparsityDetector(G, H)
    )
end
