using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

GRADIENT_TRACERS = (
    GradientTracer{BitSet},
    GradientTracer{Set{Int}},
    GradientTracer{DuplicateVector{Int}},
    GradientTracer{SortedVector{Int}},
)

include("brusselator_definition.jl")

function test_brusselator(method::AbstractSparsityDetector)
    N = 6
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    J = ADTypes.jacobian_sparsity(f!, y, x, method)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "$T" for T in GRADIENT_TRACERS
    method = TracerSparsityDetector(; gradient_tracer=T)
    test_brusselator(method)
end
