using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

include("brusselator_definition.jl")

function test_brusselator(method::AbstractSparsityDetector)
    N = 6
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    J = ADTypes.jacobian_sparsity(f!, y, x, method)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "$method" for method in (
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(DuplicateVector{UInt64}),
    TracerSparsityDetector(RecursiveSet{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
)
    test_brusselator(method)
end
