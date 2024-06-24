import SparseConnectivityTracer as SCT
using SparseConnectivityTracer
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
function test_patterns(f, x; connectivity=isone, jacobian=isone, hessian=isone)
    @testset "$f" begin
        @testset "Connecivity pattern" begin
            @test all(connectivity, connectivity_pattern(f, x))
        end
        @testset "Jacobian pattern" begin
            @test all(jacobian, jacobian_pattern(f, x))
        end
        @testset "Hessian pattern" begin
            @test all(hessian, hessian_pattern(f, x))
        end
    end
end

opnorm1(A) = opnorm(A, 1)
logabsdet_first(A) = first(logabsdet(A))
logabsdet_last(A) = last(logabsdet(A))

@testset "Scalar functions" begin
    @testset "$name" for (name, A) in TEST_MATRICES
        test_patterns(det, A)
        test_patterns(logdet, A)
        test_patterns(norm, A)
        test_patterns(eigmax, A)
        test_patterns(eigmin, A)
        test_patterns(opnorm1, A; hessian=iszero)
        test_patterns(logabsdet_first, A)
        test_patterns(logabsdet_last, A; jacobian=iszero, hessian=iszero)
    end
    @testset "`SparseMatrixCSC` (3×3)" begin
        # TODO: this is a temporary solution until sparse matrix inputs are supported (#28)
        test_patterns(A -> det(sparse(A)), rand(3, 3))
        test_patterns(A -> logdet(sparse(A)), rand(3, 3))
        test_patterns(A -> norm(sparse(A)), rand(3, 3))
        test_patterns(A -> eigmax(sparse(A)), rand(3, 3))
        test_patterns(A -> eigmin(sparse(A)), rand(3, 3))
        test_patterns(A -> opnorm1(sparse(A)), rand(3, 3); hessian=iszero)
        test_patterns(A -> logabsdet_first(sparse(A)), rand(3, 3))
        test_patterns(
            A -> logabsdet_last(sparse(A)), rand(3, 3); jacobian=iszero, hessian=iszero
        )

        test_patterns(v -> det(spdiagm(v)), rand(3))
        test_patterns(v -> logdet(spdiagm(v)), rand(3))
        test_patterns(v -> norm(spdiagm(v)), rand(3))
        test_patterns(v -> eigmax(spdiagm(v)), rand(3))
        test_patterns(v -> eigmin(spdiagm(v)), rand(3))
        test_patterns(v -> opnorm1(spdiagm(v)), rand(3); hessian=iszero)
        test_patterns(v -> logabsdet_first(spdiagm(v)), rand(3))
        test_patterns(
            v -> logabsdet_last(spdiagm(v)), rand(3); jacobian=iszero, hessian=iszero
        )
    end
end

suminv(A) = sum(inv(A))
sumexp(A) = sum(exp(A))
sumpow0(A) = sum(A^0)
sumpow1(A) = sum(A^1)
sumpow3(A) = sum(A^3)
sumpinv(A) = sum(pinv(A))

@testset "Matrix-valued functions" begin
    # Functions that only work on square matrices
    @testset "$name" for (name, A) in TEST_SQUARE_MATRICES
        test_patterns(suminv, A)
        test_patterns(sumexp, A)
        test_patterns(sumpow0, A; connectivity=iszero, jacobian=iszero, hessian=iszero)
        test_patterns(sumpow1, A; hessian=iszero)
        test_patterns(sumpow3, A)
    end
    @testset "`SparseMatrixCSC` (3×3)" begin
        # TODO: this is a temporary solution until sparse matrix inputs are supported (#28)

        test_patterns(A -> sumexp(sparse(A)), rand(3, 3))
        test_patterns(
            A -> sumpow0(sparse(A)),
            rand(3, 3);
            connectivity=iszero,
            jacobian=iszero,
            hessian=iszero,
        )
        test_patterns(A -> sumpow1(sparse(A)), rand(3, 3); hessian=iszero)
        test_patterns(A -> sumpow3(sparse(A)), rand(3, 3))

        test_patterns(v -> sumexp(spdiagm(v)), rand(3))
        test_patterns(
            v -> sumpow0(spdiagm(v)),
            rand(3);
            connectivity=iszero,
            jacobian=iszero,
            hessian=iszero,
        )
        test_patterns(v -> sumpow1(spdiagm(v)), rand(3); hessian=iszero)
        test_patterns(v -> sumpow3(spdiagm(v)), rand(3))
    end

    # Functions that work on all matrices
    @testset "$name" for (name, A) in TEST_MATRICES
        test_patterns(sumpinv, A)
    end
    @testset "`SparseMatrixCSC` (3×4)" begin
        test_patterns(A -> sumpinv(sparse(A)), rand(3, 4))
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
