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
    SUITE["Jacobian"][nameof(G)] = jacbench(TracerSparsityDetector(G))
    SUITE["Hessian"][(nameof(G), nameof(H))] = hessbench(TracerSparsityDetector(G, H))
end

run(SUITE; verbose=true)
