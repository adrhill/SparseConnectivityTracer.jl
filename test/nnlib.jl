using NNlib
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

@testset "Set type $S" for S in (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)
    x = rand(3, 3, 2, 1) # WHCN
    w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
    C = jacobian_pattern(x -> NNlib.conv(x, w), x, S)
    @test_reference "references/pattern/connectivity/NNlib/conv.txt" BitMatrix(C)
    J = jacobian_pattern(x -> NNlib.conv(x, w), x, S)
    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(J)
    @test C == J
end
