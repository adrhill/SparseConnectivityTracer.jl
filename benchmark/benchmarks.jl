using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet
using SparseConnectivityTracer:
    SimpleVectorIndexSetPattern, SimpleVectorAndMatrixIndexSetPattern

const FIRST_ORDER_PATTERNS = (
    SimpleVectorIndexSetPattern{BitSet},
    SimpleVectorIndexSetPattern{Set{Int}},
    SimpleVectorIndexSetPattern{DuplicateVector{Int}},
    SimpleVectorIndexSetPattern{RecursiveSet{Int}},
    SimpleVectorIndexSetPattern{SortedVector{Int}},
)

const SECOND_ORDER_PATTERNS = (
    SimpleVectorAndMatrixIndexSetPattern{BitSet,Set{Tuple{Int,Int}}},  #
    SimpleVectorAndMatrixIndexSetPattern{Set{Int},Set{Tuple{Int,Int}}},
    SimpleVectorAndMatrixIndexSetPattern{
        DuplicateVector{Int},DuplicateVector{Tuple{Int,Int}}
    },
    SimpleVectorAndMatrixIndexSetPattern{SortedVector{Int},SortedVector{Tuple{Int,Int}}},
    SimpleVectorAndMatrixIndexSetPattern{SortedVector{Int},Set{Tuple{Int,Int}}},
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
