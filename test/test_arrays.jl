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
function test_patterns(
    f, x; sum_out=false, connectivity=isone, jacobian=isone, hessian=isone
)
    @testset "$f" begin
        if sum_out
            _f(x) = sum(f(x))
        else
            _f = f
        end
        @testset "Connecivity pattern" begin
            @test all(connectivity, connectivity_pattern(_f, x))
        end
        @testset "Jacobian pattern" begin
            @test all(jacobian, jacobian_pattern(_f, x))
        end
        @testset "Hessian pattern" begin
            @test all(hessian, hessian_pattern(_f, x))
        end
    end
end

@testset "Scalar functions" begin
    opnorm1(A) = opnorm(A, 1)
    logabsdet_first(A) = first(logabsdet(A))
    logabsdet_last(A) = last(logabsdet(A))

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

@testset "Matrix-valued functions" begin
    pow0(A) = sum(A^0)
    pow1(A) = sum(A^1)
    pow3(A) = sum(A^3)

    # Functions that only work on square matrices
    @testset "$name" for (name, A) in TEST_SQUARE_MATRICES
        test_patterns(inv, A; sum_out=true)
        test_patterns(exp, A; sum_out=true)
        test_patterns(
            pow0, A; sum_out=true, connectivity=iszero, jacobian=iszero, hessian=iszero
        )
        test_patterns(pow1, A; sum_out=true, hessian=iszero)
        test_patterns(pow3, A; sum_out=true)
    end
    @testset "`SparseMatrixCSC` (3×3)" begin
        # TODO: this is a temporary solution until sparse matrix inputs are supported (#28)

        test_patterns(A -> exp(sparse(A)), rand(3, 3); sum_out=true)
        test_patterns(
            A -> pow0(sparse(A)),
            rand(3, 3);
            sum_out=true,
            connectivity=iszero,
            jacobian=iszero,
            hessian=iszero,
        )
        test_patterns(A -> pow1(sparse(A)), rand(3, 3); sum_out=true, hessian=iszero)
        test_patterns(A -> pow3(sparse(A)), rand(3, 3); sum_out=true)

        test_patterns(v -> exp(spdiagm(v)), rand(3); sum_out=true)
        test_patterns(
            v -> sumpow0(spdiagm(v)),
            rand(3);
            sum_out=true,
            connectivity=iszero,
            jacobian=iszero,
            hessian=iszero,
        )
        test_patterns(v -> pow1(spdiagm(v)), rand(3); sum_out=true, hessian=iszero)
        test_patterns(v -> pow3(spdiagm(v)), rand(3); sum_out=true)
    end

    # Functions that work on all matrices
    @testset "$name" for (name, A) in TEST_MATRICES
        test_patterns(pinv, A; sum_out=true)
    end
    @testset "`SparseMatrixCSC` (3×4)" begin
        test_patterns(A -> pinv(sparse(A)), rand(3, 4); sum_out=true)
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
