using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracerBenchmarks.ODE: Brusselator!
using Test

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("tracers_definitions.jl")

function test_brusselator(detector::AbstractSparsityDetector)
    N = 6
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    J = ADTypes.jacobian_sparsity(f!, y, x, detector)
    return @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "$T" for T in GRADIENT_TRACERS
    detector = TracerSparsityDetector(T)
    test_brusselator(detector)
    yield()
end
