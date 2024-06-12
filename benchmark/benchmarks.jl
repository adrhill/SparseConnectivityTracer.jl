using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet
using SparseConnectivityTracer: IndexSetVectorPattern, CombinedPattern

const FIRST_ORDER_PATTERNS = (
    IndexSetVectorPattern{BitSet},
    IndexSetVectorPattern{Set{Int}},
    IndexSetVectorPattern{DuplicateVector{Int}},
    IndexSetVectorPattern{RecursiveSet{Int}},
    IndexSetVectorPattern{SortedVector{Int}},
)

const SECOND_ORDER_PATTERNS = (
    CombinedPattern{
        IndexSetVectorPattern{BitSet},IndexSetVectorPattern{Set{Tuple{Int,Int}}}
    },#
    CombinedPattern{
        IndexSetVectorPattern{Set{Int}},IndexSetVectorPattern{Set{Tuple{Int,Int}}}
    },
    CombinedPattern{
        IndexSetVectorPattern{DuplicateVector{Int}},
        IndexSetVectorPattern{DuplicateVector{Tuple{Int,Int}}},
    },
    CombinedPattern{
        IndexSetVectorPattern{SortedVector{Int}},
        IndexSetVectorPattern{SortedVector{Tuple{Int,Int}}},
    },
    CombinedPattern{
        IndexSetVectorPattern{SortedVector{Int}},IndexSetVectorPattern{Set{Tuple{Int,Int}}}
    },
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
