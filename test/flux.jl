using ADTypes
using ADTypes: AbstractSparsityDetector
using Flux: Conv, relu
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Test

function test_flux_conv(method::AbstractSparsityDetector)
    x = rand(3, 3, 2, 1) # WHCN
    weights = reshape(
        [
            0.98367139
            0.61198703
            0.01781049
            0.79373138
            0.01697244
            0.16676845
            0.15130460
            0.17703203
        ],
        2,
        2,
        2,
        1,
    )
    bias = [0.83478294]

    layer = Conv(weights, bias) # Conv((2, 2), 2 => 1)
    J = ADTypes.jacobian_sparsity(layer, x, method)
    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(J)

    layer = Conv(weights, bias, relu)
    @test_broken J = ADTypes.jacobian_sparsity(layer, x, method)
    # @test_reference "references/pattern/jacobian/NNlib/conv_relu.txt" BitMatrix(J)
end

@testset "$method" for method in (
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(DuplicateVector{UInt64}),
    TracerSparsityDetector(RecursiveSet{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
)
    test_flux_conv(method)
end
