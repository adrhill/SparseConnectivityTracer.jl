using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet
using SparseConnectivityTracer: IndexSetVectorPattern, CombinedPattern

const FIRST_ORDER_PATTERNS = (
    IndexSetVectorPattern{Int,BitSet},
    IndexSetVectorPattern{Int,Set{Int}},
    IndexSetVectorPattern{Int,DuplicateVector{Int}},
    IndexSetVectorPattern{Int,RecursiveSet{Int}},
    IndexSetVectorPattern{Int,SortedVector{Int}},
)

const SECOND_ORDER_PATTERNS = (
    CombinedPattern{
        IndexSetVectorPattern{Int,BitSet},IndexSetMatrixPattern{Int,Set{Tuple{Int,Int}}}
    },#
    CombinedPattern{
        IndexSetVectorPattern{Int,Set{Int}},IndexSetMatrixPattern{Int,Set{Tuple{Int,Int}}}
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,DuplicateVector{Int}},
        IndexSetMatrixPattern{Int,DuplicateVector{Tuple{Int,Int}}},
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,SortedVector{Int}},
        IndexSetMatrixPattern{Int,SortedVector{Tuple{Int,Int}}},
    },
    CombinedPattern{
        IndexSetVectorPattern{Int,SortedVector{Int}},
        IndexSetMatrixPattern{Int,Set{Tuple{Int,Int}}},
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
