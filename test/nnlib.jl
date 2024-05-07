using ADTypes
using ADTypes: AbstractSparsityDetector
using NNlib
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

function test_nnlib_conv(method::AbstractSparsityDetector)
    x = rand(3, 3, 2, 1) # WHCN
    w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
    f(x) = NNlib.conv(x, w)

    J = ADTypes.jacobian_sparsity(f, x, method)
    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(J)
end

@testset "$method" for method in (
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(DuplicateVector{UInt64}),
    TracerSparsityDetector(RecursiveSet{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
)
    test_nnlib_conv(method)
end
