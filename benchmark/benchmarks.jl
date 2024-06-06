using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet
using SparseConnectivityTracer: SimpleIndexSet, SimpleSecondOrderIndexSet

const FIRST_ORDER_PATTERNS = (
    SimpleIndexSet{BitSet},
    SimpleIndexSet{Set{Int}},
    SimpleIndexSet{DuplicateVector{Int}},
    SimpleIndexSet{RecursiveSet{Int}},
    SimpleIndexSet{SortedVector{Int}},
)

const SECOND_ORDER_PATTERNS = (
    SimpleSecondOrderIndexSet{BitSet,Set{Tuple{Int,Int}}},  #
    SimpleSecondOrderIndexSet{Set{Int},Set{Tuple{Int,Int}}},
    SimpleSecondOrderIndexSet{DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}}},
    SimpleSecondOrderIndexSet{SortedVector{Int},SortedVector{Tuple{Int,Int}}},
    SimpleSecondOrderIndexSet{SortedVector{Int},Set{Tuple{Int,Int}}},
)

include("jacobian.jl")
include("hessian.jl")

SUITE = BenchmarkGroup()
for F in FIRST_ORDER_PATTERNS
    SUITE["Jacobian"]["Global"]["$typeof(F)"] = jacbench(
        TracerSparsityDetector(; first_order=F)
    )
    SUITE["Jacobian"]["Local"]["$typeof(F)"] = jacbench(
        TracerLocalSparsityDetector(; first_order=F)
    )
end
for S in SECOND_ORDER_PATTERNS
    SUITE["Hessian"]["Global"]["$typeof(S)"] = hessbench(
        TracerSparsityDetector(; second_order=S)
    )
    SUITE["Hessian"]["Local"]["$typeof(S)"] = hessbench(
        TracerLocalSparsityDetector(; second_order=S)
    )
end
