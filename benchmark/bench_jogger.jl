using Pkg
Pkg.develop(; path=joinpath(@__DIR__, "SparseConnectivityTracerBenchmarks"))

using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, HessianTracer
using SparseConnectivityTracer: IndexSetGradientPattern, IndexSetHessianPattern
using SparseConnectivityTracer: DuplicateVector, SortedVector, RecursiveSet

SET_TYPES = (BitSet, Set{Int}, DuplicateVector{Int}, RecursiveSet{Int}, SortedVector{Int})

include("jacobian.jl")
include("hessian.jl")
include("nlpmodels.jl")

suite = BenchmarkGroup()

suite["OptimizationProblems"] = optbench([:britgas])

for S1 in SET_TYPES
    S2 = Set{Tuple{Int,Int}}

    PG = IndexSetGradientPattern{Int,S1}
    PH = IndexSetHessianPattern{Int,S1,S2,false}

    G = GradientTracer{PG}
    H = HessianTracer{PH}

    suite["Jacobian"]["Global"][nameof(S1)] = jacbench(TracerSparsityDetector(G, H))
    suite["Jacobian"]["Local"][nameof(S1)] = jacbench(TracerLocalSparsityDetector(G, H))
    suite["Hessian"]["Global"][(nameof(S1), nameof(S2))] = hessbench(
        TracerSparsityDetector(G, H)
    )
    suite["Hessian"]["Local"][(nameof(S1), nameof(S2))] = hessbench(
        TracerLocalSparsityDetector(G, H)
    )
end
