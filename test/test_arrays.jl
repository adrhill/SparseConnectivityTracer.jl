import SparseConnectivityTracer as SCT
using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, IndexSetGradientPattern
using Test

using LinearAlgebra: Symmetric, Diagonal, diagind
using LinearAlgebra: det, logdet, logabsdet, norm, opnorm
using LinearAlgebra: eigen, eigmax, eigmin
using LinearAlgebra: inv, pinv
using SparseArrays: sparse, spdiagm

#=========================#
# Weird function wrappers #
#=========================#

# These print better stack traces than lambda functions.

struct SparsifyInput{F}
    f::F
end
(s::SparsifyInput)(x) = s.f(sparse(x))

struct SpdiagmifyInput{F}
    f::F
end
(s::SpdiagmifyInput)(x) = s.f(spdiagm(x))

struct SumOutputs{F}
    f::F
end
(s::SumOutputs)(x) = sum(s.f(x))
#===================#
# Testing utilities #
#===================#

method = TracerSparsityDetector()

test_all_one(A) = @test all(isone, A)
test_all_zero(A) = @test all(iszero, A)

# Short-hand for Jacobian pattern of `x -> sum(f(A))`
Jsum(f, A) = jacobian_sparsity(SumOutputs(f), A, method)
# Test whether all entries in Jacobian are zero
testJ0(f, A) = @testset "Jacobian" test_all_zero(Jsum(f, A))
# Test whether all entries in Jacobian are one where inputs were non-zero.
testJ1(f, A) = @testset "Jacobian" test_all_one(Jsum(f, A))
function testJ1(f, A::Diagonal)
    @testset "Jacobian" begin
        jac = Jsum(f, A)
        di = diagind(A)
        for (i, x) in enumerate(jac)
            if i in di
                @test isone(x)
            else
                @test iszero(x)
            end
        end
    end
end

# Short-hand for Hessian pattern of `x -> sum(f(A))`
Hsum(f, A) = hessian_sparsity(SumOutputs(f), A, method)
# Test whether all entries in Hessian are zero
testH0(f, A) = @testset "Hessian" test_all_zero(Hsum(f, A))
# Test whether all entries in Hessian are one where inputs were non-zero.
testH1(f, A) = @testset "Hessian" test_all_one(Hsum(f, A))
function testH1(f, A::Diagonal)
    @testset "Hessian" begin
        hess = Hsum(f, A)
        di = diagind(A)

        for I in CartesianIndices(A)
            i, j = Tuple(I)
            x = hess[I]

            if i in di && j in di
                @test isone(x)
            else
                @test iszero(x)
            end
        end
    end
end

#===================#
# Arrays to test on #
#===================#

mat33 = rand(3, 3)
mat34 = rand(3, 4)
sym33 = Symmetric(rand(3, 3))
dia33 = Diagonal(rand(3))

ALL_MATRICES = (mat33, mat34, sym33, dia33)
SQUARE_MATRICES = (mat33, sym33, dia33)
NONDIAG_MATRICES = (mat33, mat34, sym33)
NONDIAG_SQUARE_MATRICES = (mat33, sym33)
DIAG_MATRICES = (dia33,)
DIAG_SQUARE_MATRICES = (dia33,)

arrayname(A) = "$(typeof(A)) $(size(A))"

#=================#
# TEST START HERE #
#=================#

@testset "Scalar functions" begin
    norm1(A) = norm(A, 1)
    norm2(A) = norm(A, 2)
    norminf(A) = norm(A, Inf)
    opnorm1(A) = opnorm(A, 1)
    opnorm2(A) = opnorm(A, 2)
    opnorminf(A) = opnorm(A, Inf)
    logabsdet_first(A) = first(logabsdet(A))
    logabsdet_last(A) = last(logabsdet(A))

    @testset "det $(arrayname(A))" for A in NONDIAG_MATRICES
        testJ1(det, A)
        testH1(det, A)
    end
    @testset "det $(arrayname(A))" for A in DIAG_MATRICES
        @test Jsum(det, A) == [1 0 0 0 1 0 0 0 1;]
        @test Hsum(det, A) == [
            0  0  0  0  1  0  0  0  1
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            1  0  0  0  0  0  0  0  1
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            1  0  0  0  1  0  0  0  0
        ]
    end
    @testset "logdet $(arrayname(A))" for A in ALL_MATRICES
        testJ1(logdet, A)
        testH1(logdet, A)
    end
    @testset "norm(A, 1) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(norm1, A)
        testH0(norm1, A)
    end
    @testset "norm(A, 2) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(norm2, A)
        testH1(norm2, A)
    end
    @testset "norm(A, Inf) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(norminf, A)
        testH0(norminf, A)
    end
    @testset "eigmax $(arrayname(A))" for A in ALL_MATRICES
        testJ1(eigmax, A)
        testH1(eigmax, A)
    end
    @testset "eigmin $(arrayname(A))" for A in ALL_MATRICES
        testJ1(eigmin, A)
        testH1(eigmin, A)
    end
    @testset "opnorm(A, 1) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(opnorm1, A)
        testH0(opnorm1, A)
    end
    @testset "opnorm(A, 2) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(opnorm2, A)
        testH1(opnorm2, A)
    end
    @testset "opnorm(A, Inf) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(opnorminf, A)
        testH0(opnorminf, A)
    end
    @testset "first(logabsdet(A)) $(arrayname(A))" for A in ALL_MATRICES
        testJ1(logabsdet_first, A)
        testH1(logabsdet_first, A)
    end
    @testset "last(logabsdet(A)) $(arrayname(A))" for A in ALL_MATRICES
        testJ0(logabsdet_last, A)
        testH0(logabsdet_last, A)
    end

    if VERSION >= v"1.9"
        @testset "`SparseMatrixCSC` (3×3)" begin
            A = rand(3, 3)
            v = rand(3)

            # TODO: this is a temporary solution until sparse matrix inputs are supported (#28)
            @testset "det" begin
                testJ1(SparsifyInput(det), A)
                testH1(SparsifyInput(det), A)
                testJ1(SpdiagmifyInput(det), v)
                testH1(SpdiagmifyInput(det), v)
            end
            @testset "logdet" begin
                testJ1(SparsifyInput(logdet), A)
                testH1(SparsifyInput(logdet), A)
                testJ1(SpdiagmifyInput(logdet), v)
                testH1(SpdiagmifyInput(logdet), v)
            end
            @testset "norm" begin
                testJ1(SparsifyInput(norm), A)
                testH1(SparsifyInput(norm), A)
                testJ1(SpdiagmifyInput(norm), v)
                testH1(SpdiagmifyInput(norm), v)
            end
            @testset "eigmax" begin
                testJ1(SparsifyInput(eigmax), A)
                testH1(SparsifyInput(eigmax), A)
                testJ1(SpdiagmifyInput(eigmax), v)
                testH1(SpdiagmifyInput(eigmax), v)
            end
            @testset "eigmin" begin
                testJ1(SparsifyInput(eigmin), A)
                testH1(SparsifyInput(eigmin), A)
                testJ1(SpdiagmifyInput(eigmin), v)
                testH1(SpdiagmifyInput(eigmin), v)
            end
            @testset "opnorm(x, 1)" begin
                testJ1(SparsifyInput(opnorm1), A)
                testH0(SparsifyInput(opnorm1), A)
                testJ1(SpdiagmifyInput(opnorm1), v)
                testH0(SpdiagmifyInput(opnorm1), v)
            end
            @testset "first(logabsdet(x))" begin
                testJ1(SparsifyInput(logabsdet_first), A)
                testH1(SparsifyInput(logabsdet_first), A)
                testJ1(SpdiagmifyInput(logabsdet_first), v)
                testH1(SpdiagmifyInput(logabsdet_first), v)
            end
            @testset "last(logabsdet(x))" begin
                testJ0(SparsifyInput(logabsdet_last), A)
                testH0(SparsifyInput(logabsdet_last), A)
                testJ0(SpdiagmifyInput(logabsdet_last), v)
                testH0(SpdiagmifyInput(logabsdet_last), v)
            end
        end
    end
end

@testset "Matrix-valued functions" begin
    pow0(A) = A^0
    pow3(A) = A^3

    # Functions that only work on square matrices
    @testset "inv $(arrayname(A))" for A in NONDIAG_SQUARE_MATRICES
        testJ1(inv, A)
        testH1(inv, A)
    end
    @testset "inv $(arrayname(A))" for A in DIAG_SQUARE_MATRICES
        @test Hsum(inv, A) == [
            1  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  1  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  1
        ]
    end
    @testset "exp $(arrayname(A))" for A in NONDIAG_SQUARE_MATRICES
        testJ1(exp, A)
        testH1(exp, A)
    end
    @testset "exp $(arrayname(A))" for A in DIAG_SQUARE_MATRICES
        testJ1(exp, A)
        @test Hsum(exp, A) == [
            1  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  1  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  1
        ]
    end
    @testset "pow0 $(arrayname(A))" for A in NONDIAG_SQUARE_MATRICES
        testJ0(pow0, A)
        testH0(pow0, A)
    end
    @testset "pow0 $(arrayname(A))" for A in DIAG_SQUARE_MATRICES
        # TODO: these should be zero and are currently too conservative
        @test_broken all(iszero, Jsum(pow0, A))
        @test_broken all(iszero, Hsum(pow0, A))
    end
    @testset "pow3 $(arrayname(A))" for A in SQUARE_MATRICES
        testJ1(pow3, A)
        testH1(pow3, A)
    end

    if VERSION >= v"1.9"
        A = rand(3, 3)
        v = rand(3)

        @testset "`SparseMatrixCSC` (3×3)" begin
            # TODO: this is a temporary solution until sparse matrix inputs are supported (#28)

            testJ1(SparsifyInput(exp), A)
            testH1(SparsifyInput(exp), A)

            testJ0(SparsifyInput(pow0), A)
            testH0(SparsifyInput(pow0), A)

            testJ1(SparsifyInput(pow3), A)
            testH1(SparsifyInput(pow3), A)

            testJ1(SpdiagmifyInput(exp), v)
            testH1(SpdiagmifyInput(exp), v)

            if VERSION >= v"1.10"
                # issue with custom _mapreducezeros in SparseArrays on Julia 1.6
                testJ0(SpdiagmifyInput(pow0), v)
                testH0(SpdiagmifyInput(pow0), v)

                testJ1(SpdiagmifyInput(pow3), v)
                testH1(SpdiagmifyInput(pow3), v)
            end
        end
    end

    # Functions that work on all matrices
    @testset "pinv $(arrayname(A))" for A in ALL_MATRICES
        testJ1(pinv, A)
        testH1(pinv, A)
    end
    if VERSION >= v"1.9"
        @testset "`SparseMatrixCSC` (3×4)" begin
            testJ1(SparsifyInput(pinv), rand(3, 4))
            testH1(SparsifyInput(pinv), rand(3, 4))
        end
    end
end

S = BitSet
P = IndexSetGradientPattern{Int,S}
TG = GradientTracer{P}

@testset "Matrix division" begin
    t1 = TG(P(S([1, 3, 4])))
    t2 = TG(P(S([2, 4])))
    t3 = TG(P(S([8, 9])))
    t4 = TG(P(S([8, 9])))
    A = [t1 t2; t3 t4]
    s_out = S([1, 2, 3, 4, 8, 9])

    x = rand(2)
    b = A \ x
    @test all(t -> SCT.gradient(t) == s_out, b)
end

@testset "Eigenvalues" begin
    t1 = TG(P(S([1, 3, 4])))
    t2 = TG(P(S([2, 4])))
    t3 = TG(P(S([8, 9])))
    t4 = TG(P(S([8, 9])))
    A = [t1 t2; t3 t4]
    s_out = S([1, 2, 3, 4, 8, 9])
    values, vectors = eigen(A)
    @test size(values) == (2,)
    @test size(vectors) == (2, 2)
    @test all(t -> SCT.gradient(t) == s_out, values)
    @test all(t -> SCT.gradient(t) == s_out, vectors)
end

if VERSION >= v"1.9"
    @testset "SparseMatrixCSC construction" begin
        t1 = TG(P(S(1)))
        t2 = TG(P(S(2)))
        t3 = TG(P(S(3)))
        SA = sparse([t1 t2; t3 0])
        @test length(SA.nzval) == 3

        res = opnorm(SA, 1)
        @test SCT.gradient(res) == S([1, 2, 3])
    end
end
