using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet

SET_TYPES = (BitSet, Set{Int}, DuplicateVector{Int}, RecursiveSet{Int}, SortedVector{Int})

include("jacobian.jl")
include("hessian.jl")
include("nlpmodels.jl")

SUITE = BenchmarkGroup()
#=  # TODO: uncomment
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
=#

SUITE["OptimizationProblems"] = optbench()
