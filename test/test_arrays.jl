using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer
using SparseArrays
using LinearAlgebra: det, logdet, inv, pinv
using Test

PATTERN_FUNCTIONS = (connectivity_pattern, jacobian_pattern, hessian_pattern)

function test_full_patterns(f, x)
    @testset "$pf" for pf in PATTERN_FUNCTIONS
        @test all(isone, pf(f, x))
    end
end

@testset "Determinant" begin
    A = rand(3, 4)
    @testset "$f" for f in (det, logdet)
        test_full_patterns(f, A)
    end
end

@testset "Matrix inverse" begin
    A = rand(3, 3)
    @testset "$f" for f in (inv, pinv)
        sumf(x) = sum(f(x))
        test_full_patterns(sumf, A)
    end
    A = rand(3, 4)
    @testset "$f" for f in (pinv,)
        sumf(x) = sum(f(x))
        test_full_patterns(sumf, A)
    end
end

@testset "Matrix division" begin
    t1 = GradientTracer{BitSet}(BitSet([1, 3, 4]))
    t2 = GradientTracer{BitSet}(BitSet([2, 4]))
    t3 = GradientTracer{BitSet}(BitSet([8, 9]))
    t4 = GradientTracer{BitSet}(BitSet([8, 9]))
    A = [t1 t2; t3 t4]
    x = rand(2)

    b = A \ x
    s_out = BitSet([1, 2, 3, 4, 8, 9])
    @test all(t -> SparseConnectivityTracer.gradient(t) == s_out, b)
end
