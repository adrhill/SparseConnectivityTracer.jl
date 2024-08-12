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
PH = DictHessianPattern{Int,BitSet,Dict{Int,BitSet},Shared}
H = HessianTracer{PH}
method = TracerSparsityDetector(; hessian_tracer_type=H)
suite["Hessian"]["Global (shared)"] = hessbench(method)
