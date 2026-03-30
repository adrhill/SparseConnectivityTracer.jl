using SparseConnectivityTracer
using SparseConnectivityTracer: FixedSizeBitSet
import SparseConnectivityTracer as SCT
using Test
using JLArrays

@testset "FixedSizeBitSet" begin
    @test SCT.bitwidth(UInt16) == 16
    @test SCT.bitwidth(UInt32) == 32
    @test SCT.bitwidth(UInt64) == 64

    S = FixedSizeBitSet{UInt16, 3}
    @test_throws ArgumentError S(0)
    @test collect(S(1)) == [1]
    @test collect(S(10)) == [10]
    @test collect(S(1) ∪ S(2)) == [1, 2]
    @test collect(S(1) ∪ S(2) ∪ S(3)) == [1, 2, 3]
    @test collect(S(1) ∪ S(1) ∪ S(2)) == [1, 2]
    @test collect(S(48)) == [48]
    @test_throws ArgumentError S(49)

    @test length(S()) == 0
    @test length(S(1)) == 1
    @test length(S(1) ∪ S(10)) == 2
end

@testset "Jacobian sparsity" begin
    detector_ref = TracerSparsityDetector()
    @testset for (I, N) in Iterators.product(Type[UInt8, UInt16, UInt32, UInt64], [2, 5, 8])
        detector = TracerSparsityDetector(; gradient_pattern_type = FixedSizeBitSet{I, N})
        for x in (rand(4), rand(40), rand(400))
            @test jacobian_sparsity(diff, x, detector) ==
                jacobian_sparsity(diff, x, detector_ref)
        end
    end
end;

@testset "GPU compat" begin
    x = jl(rand(3))
    @test_throws ArgumentError jacobian_sparsity(diff, x, TracerSparsityDetector())
    detector = TracerSparsityDetector(; gradient_pattern_type = FixedSizeBitSet{UInt8, 1})
    @test jacobian_sparsity(diff, x, detector) ==
        jacobian_sparsity(diff, Vector(x), TracerSparsityDetector())
end
