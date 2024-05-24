using BenchmarkTools
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet

SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)

include("jacobian.jl")
include("hessian.jl")

SUITE = BenchmarkGroup()
for S in SET_TYPES
    SUITE["Jacobian"][nameof(S)] = jacbench(S)
    SUITE["Hessian"][nameof(S)] = hessbench(S)
end
