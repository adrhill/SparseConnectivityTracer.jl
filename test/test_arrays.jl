using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer
using SparseArrays
using LinearAlgebra: det, logdet, logabsdet
using LinearAlgebra: inv, pinv
using Test

PATTERN_FUNCTIONS = (connectivity_pattern, jacobian_pattern, hessian_pattern)

TEST_SQUARE_MATRICES = Dict(
    "Matrix (3×3)" => rand(3, 3),
    "Symmetric (3×3)" => Symmetric(rand(3, 3)),
    "Diagonal (3×3)" => Diagonal(rand(3)),
)
TEST_MATRICES = merge(TEST_SQUARE_MATRICES, Dict("Matrix (3×4)" => rand(3, 4)))

function test_full_patterns(f, x)
    @testset "$f_pattern" for f_pattern in PATTERN_FUNCTIONS
        @test all(isone, f_pattern(f, x))
    end
end

@testset "Scalar functions" begin
    @testset "$f" for f in (det, logdet, logabsdet, norm, opnorm, eigmax, eigmin)
        @testset "$name" for (name, A) in TEST_MATRICES
            test_full_patterns(f, A)
        end
    end
end

@testset "Matrix-valued functions" begin
    # Functions that only work on square matrices
    @testset "$f" for f in (inv, exp, A -> A^3)
        sumf(x) = sum(f(x))
        @testset "$name" for (name, A) in TEST_SQUARE_MATRICES
            test_full_patterns(sumf, A)
        end
    end
    # Functions that work on all matrices
    @testset "$f" for f in (pinv,)
        sumf(x) = sum(f(x))
        @testset "$name" for (name, A) in TEST_MATRICES
            test_full_patterns(sumf, A)
        end
    end
end

t1 = GradientTracer{BitSet}(BitSet([1, 3, 4]))
t2 = GradientTracer{BitSet}(BitSet([2, 4]))
t3 = GradientTracer{BitSet}(BitSet([8, 9]))
t4 = GradientTracer{BitSet}(BitSet([8, 9]))
A = [t1 t2; t3 t4]
s_out = BitSet([1, 2, 3, 4, 8, 9])

@testset "Matrix division" begin
    x = rand(2)
    b = A \ x
    @test all(t -> SparseConnectivityTracer.gradient(t) == s_out, b)
end
@testset "Eigenvalues" begin
    values, vectors = eigen(A)
    @test size(values) == (2,)
    @test size(vectors) == (2, 2)
    @test all(t -> SparseConnectivityTracer.gradient(t) == s_out, values)
    @test all(t -> SparseConnectivityTracer.gradient(t) == s_out, vectors)
end
