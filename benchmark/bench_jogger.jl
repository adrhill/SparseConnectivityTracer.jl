using Pkg
Pkg.develop(; path=joinpath(@__DIR__, "SparseConnectivityTracerBenchmarks"))

using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: HessianTracer, DictHessianPattern, Shared

include("jacobian.jl")
include("hessian.jl")
include("nlpmodels.jl")

suite = BenchmarkGroup()
suite["OptimizationProblems"] = optbench([:britgas])

suite["Jacobian"]["Global"] = jacbench(TracerSparsityDetector())
suite["Jacobian"]["Local"] = jacbench(TracerLocalSparsityDetector())
suite["Hessian"]["Global"] = hessbench(TracerSparsityDetector())
suite["Hessian"]["Local"] = hessbench(TracerLocalSparsityDetector())

# Shared tracers 
P = DictHessianPattern{Int,BitSet,Dict{Int,BitSet},Shared}
H = HessianTracer{P}
suite["Hessian"]["Global shared"] = hessbench(
    TracerSparsityDetector(; hessian_tracer_type=H)
)
suite["Hessian"]["Local shared"] = hessbench(
    TracerLocalSparsityDetector(; hessian_tracer_type=H)
)
