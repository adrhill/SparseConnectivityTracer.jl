import SparseConnectivityTracer as SCT
using SparseConnectivityTracer: GradientTracer
using LinearAlgebra: Symmetric, Diagonal
using LinearAlgebra: det, logdet, logabsdet, norm, opnorm
using LinearAlgebra: eigen, eigmax, eigmin
using LinearAlgebra: inv, pinv
using SparseArrays: sparse, spdiagm
using Test

PATTERN_FUNCTIONS = (connectivity_pattern, jacobian_pattern, hessian_pattern)

TEST_SQUARE_MATRICES = Dict(
    "`Matrix` (3×3)" => rand(3, 3),
    "`Symmetric` (3×3)" => Symmetric(rand(3, 3)),
    "`Diagonal` (3×3)" => Diagonal(rand(3)),
)
TEST_MATRICES = merge(TEST_SQUARE_MATRICES, Dict("`Matrix` (3×4)" => rand(3, 4)))

S = BitSet
TG = GradientTracer{S}

# NOTE: we currently test for conservative patterns on array overloads
# Changes making array overloads less convervative will break these tests, but are welcome!  
function test_full_patterns(f, x)
    @testset "$f_pattern" for f_pattern in PATTERN_FUNCTIONS
        @test all(isone, f_pattern(f, x))
    end
end

@testset "Scalar functions" begin
    @testset "$f" for f in (det, logdet, logabsdet, norm, eigmax, eigmin)
        @testset "$name" for (name, A) in TEST_MATRICES
            test_full_patterns(f, A)
        end
        @testset "`SparseMatrixCSC` (3×3)" begin
            test_full_patterns(A -> f(sparse(A)), rand(3, 3))
            test_full_patterns(A -> f(spdiagm(A)), rand(3))
        end
    end
    @testset "opnorm" begin
        @testset "$name" for (name, A) in TEST_MATRICES
            @test all(isone, connectivity_pattern(a -> opnorm(a, 1), A))
            @test all(isone, jacobian_pattern(a -> opnorm(a, 1), A))
            @test all(iszero, hessian_pattern(a -> opnorm(a, 1), A))
        end
        @testset "`SparseMatrixCSC` (3×3)" begin
            @test all(isone, connectivity_pattern(a -> opnorm(sparse(a), 1), rand(3, 3)))
            @test all(isone, jacobian_pattern(a -> opnorm(sparse(a), 1), rand(3, 3)))
            @test all(iszero, hessian_pattern(a -> opnorm(sparse(a), 1), rand(3, 3)))

            @test all(isone, connectivity_pattern(a -> opnorm(spdiagm(a), 1), rand(3)))
            @test all(isone, jacobian_pattern(a -> opnorm(spdiagm(a), 1), rand(3)))
            @test all(iszero, hessian_pattern(a -> opnorm(spdiagm(a), 1), rand(3)))
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
    @testset "$f" for f in (exp, A -> A^3)
        sumf(x) = sum(f(x))
        @testset "`SparseMatrixCSC` (3×3)" begin
            test_full_patterns(A -> sumf(sparse(A)), rand(3, 3))
        end
    end
    # Functions that work on all matrices
    @testset "$f" for f in (pinv,)
        sumf(x) = sum(f(x))
        @testset "$name" for (name, A) in TEST_MATRICES
            test_full_patterns(sumf, A)
        end
        @testset "`SparseMatrixCSC` (3×4)" begin
            test_full_patterns(A -> sumf(sparse(A)), rand(3, 4))
        end
    end
end

@testset "Matrix division" begin
    t1 = TG(S([1, 3, 4]))
    t2 = TG(S([2, 4]))
    t3 = TG(S([8, 9]))
    t4 = TG(S([8, 9]))
    A = [t1 t2; t3 t4]
    s_out = S([1, 2, 3, 4, 8, 9])

    x = rand(2)
    b = A \ x
    @test all(t -> SCT.gradient(t) == s_out, b)
end

@testset "Eigenvalues" begin
    t1 = TG(S([1, 3, 4]))
    t2 = TG(S([2, 4]))
    t3 = TG(S([8, 9]))
    t4 = TG(S([8, 9]))
    A = [t1 t2; t3 t4]
    s_out = S([1, 2, 3, 4, 8, 9])
    values, vectors = eigen(A)
    @test size(values) == (2,)
    @test size(vectors) == (2, 2)
    @test all(t -> SCT.gradient(t) == s_out, values)
    @test all(t -> SCT.gradient(t) == s_out, vectors)
end

@testset "SparseMatrixCSC construction" begin
    t1 = TG(S(1))
    t2 = TG(S(2))
    t3 = TG(S(3))
    SA = sparse([t1 t2; t3 0])
    @test length(SA.nzval) == 3

    res = opnorm(SA, 1)
    @test SCT.gradient(res) == S([1, 2, 3])
end
