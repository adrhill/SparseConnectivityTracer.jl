using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet

SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)

include("jacobian.jl")
include("hessian.jl")

SUITE = BenchmarkGroup()
for G in SET_TYPES
    H = Set{Tuple{Int,Int}}
    SUITE["Jacobian"]["Global"][nameof(G)] = jacbench(TracerSparsityDetector(G))
    SUITE["Jacobian"]["Local"][nameof(G)] = jacbench(TracerLocalSparsityDetector(G))
    SUITE["Hessian"]["Global"][(nameof(G), nameof(H))] = hessbench(
        TracerSparsityDetector(G, H)
    )
    SUITE["Hessian"]["Local"][(nameof(G), nameof(H))] = hessbench(
        TracerLocalSparsityDetector(G, H)
    )
end
