using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using SparseConnectivityTracerBenchmarks.ODE: Brusselator!
using Test

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("tracers_definitions.jl")

function test_brusselator(method::AbstractSparsityDetector)
    N = 6
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    J = ADTypes.jacobian_sparsity(f!, y, x, method)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "$T" for T in GRADIENT_TRACERS
    method = TracerSparsityDetector(; gradient_tracer_type=T)
    test_brusselator(method)
    yield()
end
