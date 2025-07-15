import SparseConnectivityTracer as SCT
using SparseConnectivityTracer
using SparseConnectivityTracer:
    GradientTracer,
    Dual,
    isemptytracer,
    MissingPrimalError,
    split_dual_array,
    tracer
using Test

using LinearAlgebra: Symmetric, Diagonal, diagind
using LinearAlgebra: det, logdet, logabsdet, norm, opnorm
using LinearAlgebra: eigen, eigmax, eigmin
using LinearAlgebra: inv, pinv, dot
using SparseArrays: sparse, spdiagm

S = BitSet
TG = GradientTracer{eltype(S), S}

# Utilities for quick testing
idx2set(is) = S(is)
idx2set(r::AbstractRange) = idx2set(collect(r))
idx2tracer(is) = TG(idx2set(is))

function sameidx(s1::AbstractSet, s2::AbstractSet)
    same = s1 == s2
    if same
        return true
    else
        println("Index sets don't match:")
        println("Detected:  ", s1)
        println("Reference: ", s2)
        return false
    end
end

function sameidx(t1::T, t2::T) where {T <: GradientTracer}
    return sameidx(SCT.gradient(t1), SCT.gradient(t2))
end
sameidx(t::GradientTracer, s::AbstractSet) = sameidx(SCT.gradient(t), s)
sameidx(t::GradientTracer, i) = sameidx(t, idx2set(i))

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

norm1(A) = norm(A, 1)
norm2(A) = norm(A, 2)
norminf(A) = norm(A, Inf)
opnorm1(A) = opnorm(A, 1)
opnorm2(A) = opnorm(A, 2)
opnorminf(A) = opnorm(A, Inf)
logabsdet_first(A) = first(logabsdet(A))
logabsdet_last(A) = last(logabsdet(A))
pow0(A) = A^0
pow3(A) = A^3

#===================#
# Testing utilities #
#===================#

detector = TracerSparsityDetector()

allone(A) = all(isone, A)
allzero(A) = all(iszero, A)

# Short-hand for Jacobian pattern of `x -> sum(f(A))`
Jsum(f, A) = jacobian_sparsity(SumOutputs(f), A, detector)
# Test whether all entries in Jacobian are zero
testJ0(f, A) = @testset "Jacobian" begin
    @test allzero(Jsum(f, A))
end
# Test whether all entries in Jacobian are one where inputs were non-zero.
testJ1(f, A) = @testset "Jacobian" begin
    @test allone(Jsum(f, A))
end
function testJ1(f, A::Diagonal)
    return @testset "Jacobian" begin
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
Hsum(f, A) = hessian_sparsity(SumOutputs(f), A, detector)
# Test whether all entries in Hessian are zero
testH0(f, A) = @testset "Hessian" begin
    @test allzero(Hsum(f, A))
end
# Test whether all entries in Hessian are one where inputs were non-zero.
testH1(f, A) = @testset "Hessian" begin
    @test allone(Hsum(f, A))
end
function testH1(f, A::Diagonal)
    return @testset "Hessian" begin
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

@testset "Matrix-valued functions" begin
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
    @testset "pow0 $(arrayname(A))" for A in SQUARE_MATRICES
        testJ0(pow0, A)
        testH0(pow0, A)
    end
    @testset "pow3 $(arrayname(A))" for A in SQUARE_MATRICES
        testJ1(pow3, A)
        testH1(pow3, A)
    end

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

        testJ0(SpdiagmifyInput(pow0), v)
        testH0(SpdiagmifyInput(pow0), v)

        testJ1(SpdiagmifyInput(pow3), v)
        testH1(SpdiagmifyInput(pow3), v)
    end

    # Functions that work on all matrices
    @testset "pinv $(arrayname(A))" for A in ALL_MATRICES
        testJ1(pinv, A)
        testH1(pinv, A)
    end
    @testset "`SparseMatrixCSC` (3×4)" begin
        testJ1(SparsifyInput(pinv), rand(3, 4))
        testH1(SparsifyInput(pinv), rand(3, 4))
    end
end

@testset "clamp!" begin
    t1 = idx2tracer(1)
    t2 = idx2tracer(2)
    t3 = idx2tracer(3)
    t4 = idx2tracer(4)
    A = [t1 t2; t3 t4]

    t_lo = idx2tracer(5)
    t_hi = idx2tracer(6)

    out = clamp!(A, 0.0, 1.0)
    @test sameidx(out[1, 1], 1)
    @test sameidx(out[1, 2], 2)
    @test sameidx(out[2, 1], 3)
    @test sameidx(out[2, 2], 4)

    out = clamp!(A, t_lo, 1.0)
    @test sameidx(out[1, 1], [1, 5])
    @test sameidx(out[1, 2], [2, 5])
    @test sameidx(out[2, 1], [3, 5])
    @test sameidx(out[2, 2], [4, 5])

    out = clamp!(A, 0.0, t_hi)
    @test sameidx(out[1, 1], [1, 6])
    @test sameidx(out[1, 2], [2, 6])
    @test sameidx(out[2, 1], [3, 6])
    @test sameidx(out[2, 2], [4, 6])

    out = clamp!(A, t_lo, t_hi)
    @test sameidx(out[1, 1], [1, 5, 6])
    @test sameidx(out[1, 2], [2, 5, 6])
    @test sameidx(out[2, 1], [3, 5, 6])
    @test sameidx(out[2, 2], [4, 5, 6])
end

@testset "Matrix multiplication" begin
    t1 = idx2tracer([1])
    t2 = idx2tracer([2])
    t3 = idx2tracer([3])
    t4 = idx2tracer([4])
    t5 = idx2tracer([5])
    t6 = idx2tracer([6])
    t7 = idx2tracer([7])
    t8 = idx2tracer([8])
    t9 = idx2tracer([9])

    A_t = [
        t1 t2
        t3 t4
        t5 t6
    ]
    A_p = rand(3, 2)

    x_t = [t7, t8]
    y_t = [t7, t9]

    x_p = rand(2)
    y_p = rand(2)

    @testset "Global" begin
        b_pp = A_p * x_p
        @testset "Tracer-Primal" begin
            b_tp = A_t * x_p
            @test size(b_tp) == size(b_pp)
            @test all(
                map(
                    sameidx,
                    b_tp,
                    [
                        t1 + t2
                        t3 + t4
                        t5 + t6
                    ],
                )
            )
            B_tp = A_t * hcat(x_p, y_p)
            @test size(B_tp) == (3, 2)
            @test all(
                map(
                    sameidx,
                    B_tp,
                    [
                        (t1 + t2)       (t1 + t2)
                        (t3 + t4)       (t3 + t4)
                        (t5 + t6)       (t5 + t6)
                    ],
                )
            )
            @test_throws DimensionMismatch A_t * vcat(x_p, x_p)
            @test_throws DimensionMismatch A_t * hcat(x_p[1:1], x_p[1:1])
        end
        @testset "Primal-Tracer" begin
            b_pt = A_p * x_t
            @test size(b_pt) == size(b_pp)
            @test all(
                map(
                    sameidx,
                    b_pt,
                    [
                        t7 + t8
                        t7 + t8
                        t7 + t8
                    ],
                )
            )
            B_pt = A_p * hcat(x_t, y_t)
            @test size(B_pt) == (3, 2)
            @test all(
                map(
                    sameidx,
                    B_pt,
                    [
                        (t7 + t8)         (t7 + t9)
                        (t7 + t8)         (t7 + t9)
                        (t7 + t8)         (t7 + t9)
                    ],
                )
            )
            @test_throws DimensionMismatch A_p * vcat(x_t, x_t)
            @test_throws DimensionMismatch A_p * hcat(x_t[1:1], x_t[1:1])
        end
        @testset "Tracer-Tracer" begin
            b_tt = A_t * x_t
            @test size(b_tt) == size(b_pp)
            @test all(
                map(
                    sameidx,
                    b_tt,
                    [
                        t1 + t2 + t7 + t8
                        t3 + t4 + t7 + t8
                        t5 + t6 + t7 + t8
                    ],
                )
            )
            B_tt = A_t * hcat(x_t, y_t)
            @test size(B_tt) == (3, 2)
            @test all(
                map(
                    sameidx,
                    B_tt,
                    [
                        (t1 + t2 + t7 + t8)           (t1 + t2 + t7 + t9)
                        (t3 + t4 + t7 + t8)           (t3 + t4 + t7 + t9)
                        (t5 + t6 + t7 + t8)           (t5 + t6 + t7 + t9)
                    ],
                ),
            )
            @test_throws DimensionMismatch A_t * vcat(x_t, x_t)
            @test_throws DimensionMismatch A_t * hcat(x_t[1:1], x_t[1:1])
        end
    end
end

@testset "Matrix division" begin
    t1 = idx2tracer([1, 3, 4])
    t2 = idx2tracer([2, 4])
    t3 = idx2tracer([8, 9])
    t4 = idx2tracer([8, 9])
    A_t = [t1 t2; t3 t4]
    A_p = rand(2, 2)

    t5 = idx2tracer([6])
    t6 = idx2tracer([5, 7])
    x_t = [t5; t6]
    x_p = rand(2)

    set_tp = set_A = idx2set([1, 2, 3, 4, 8, 9])
    set_pt = set_x = idx2set([5, 6, 7])
    set_tt = union(set_tp, set_pt)

    @testset "Global" begin
        b_pp = A_p \ x_p
        @testset "Tracer-Primal" begin
            b_tp = A_t \ x_p
            @test size(b_tp) == size(b_pp)
            @test all(t -> sameidx(t, set_tp), b_tp)
        end
        @testset "Primal-Tracer" begin
            b_pt = A_p \ x_t
            @test size(b_pt) == size(b_pp)
            @test all(t -> sameidx(t, set_pt), b_pt)
        end
        @testset "Tracer-Tracer" begin
            b_tt = A_t \ x_t
            @test size(b_tt) == size(b_pp)
            @test all(t -> sameidx(t, set_tt), b_tt)
        end
    end
    @testset "Local" begin
        @testset "$P" for P in (Float32, BigFloat)
            # https://github.com/adrhill/SparseConnectivityTracer.jl/issues/235

            A_p = rand(P, 2, 2)
            A_d = Dual.(A_p, A_t)
            @test size(A_d) == size(A_p)

            x_p = rand(P, 2)
            x_d = Dual.(x_p, x_t)
            @test size(x_d) == size(x_p)

            b_pp = A_p \ x_p

            @testset "Dual-Primal" begin
                b_dp = A_d \ x_p
                primals_dp, _ = split_dual_array(b_dp)
                @test size(b_dp) == size(b_pp)
                @test primals_dp == b_pp
                @test all(d -> sameidx(tracer(d), set_tp), b_dp)
            end
            @testset "Primal-Dual" begin
                b_pd = A_p \ x_d
                primals_pd, _ = split_dual_array(b_pd)
                @test size(b_pd) == size(b_pp)
                @test primals_pd == b_pp
                @test all(d -> sameidx(tracer(d), set_pt), b_pd)
            end
            @testset "Dual-Dual" begin
                b_dd = A_d \ x_d
                primals_dd, _ = split_dual_array(b_dd)
                @test size(b_dd) == size(b_pp)
                @test primals_dd == b_pp
                @test all(d -> sameidx(tracer(d), set_tt), b_dd)
            end
        end
    end
end

@testset "Dot" begin
    t1 = idx2tracer([1, 3, 4])
    t2 = idx2tracer([2, 4])
    t3 = idx2tracer([5, 6])
    t4 = idx2tracer([7, 8])

    tx = [t1, t2]
    ty = [t3, t4]
    tA = [idx2tracer(9) idx2tracer(10); idx2tracer(11) idx2tracer(12)]

    # SubArray variants of tx and ty
    B = [t1 t2; t3 t4]
    subx = view(B, 1, :)
    suby = view(B, 2, :)

    rx = rand(2)
    ry = rand(2)
    rA = rand(2, 2)
    sx = sparse([1; 0])
    sy = sparse([0; 1])
    sA = sparse([1 0; 0 2])

    @testset "scalar-scalar" begin
        @test sameidx(dot(t1, t2), 1:4)
        @test sameidx(dot(t1, t3), [1, 3, 4, 5, 6])
        @test sameidx(dot(t1, 1.0), [1, 3, 4])
        @test sameidx(dot(1.0, t2), [2, 4])
    end
    @testset "vector-vector" begin
        @test sameidx(dot(tx, ty), 1:8)
        @test sameidx(dot(tx, ry), 1:4)
        @test sameidx(dot(rx, ty), 5:8)

        @testset "SubArrays" begin
            @test subx isa SubArray
            @test suby isa SubArray
            @test sameidx(dot(subx, suby), 1:8)
            @test sameidx(dot(subx, ry), 1:4)
            @test sameidx(dot(rx, suby), 5:8)
        end

        @testset "SparseArrays" begin
            @test sameidx(dot(tx, sy), [2, 4])
            @test sameidx(dot(sx, ty), 5:6)
        end
        @test_throws DimensionMismatch dot(tx, rand(3))
        @test_throws DimensionMismatch dot(rand(3), ty)

        txe = TG[]
        tye = TG[]
        out = dot(txe, tye)
        @test isemptytracer(out)
    end
    @testset "vector-Matrix-vector" begin
        @test sameidx(dot(tx, rA, ry), 1:4)
        @test sameidx(dot(tx, tA, ry), vcat(1:4, 9:12))
        @test sameidx(dot(tx, rA, ty), 1:8)
        @test sameidx(dot(tx, tA, ty), 1:12)
        @test sameidx(dot(rx, tA, ty), 5:12)
        @test sameidx(dot(rx, rA, ty), 5:8)
        @test sameidx(dot(rx, tA, ry), 9:12)

        @testset "SubArrays" begin
            @test subx isa SubArray
            @test suby isa SubArray
            @test sameidx(dot(subx, rA, ry), 1:4)
            @test sameidx(dot(subx, tA, ry), vcat(1:4, 9:12))
            @test sameidx(dot(subx, rA, suby), 1:8)
            @test sameidx(dot(subx, tA, suby), 1:12)
            @test sameidx(dot(rx, tA, suby), 5:12)
            @test sameidx(dot(rx, rA, suby), 5:8)
            @test sameidx(dot(rx, tA, ry), 9:12)
        end

        @testset "SparseArrays" begin
            # Some tests are broken since there is no specialized support for SparseArrays.
            # The purpose of these tests is to catch ambiguity errors.
            @test_nowarn dot(tx, sA, ry)
            @test_nowarn dot(tx, rA, sy)
            @test_nowarn dot(tx, sA, sy)
            @test_nowarn dot(tx, tA, sy)
            @test_nowarn dot(tx, sA, ty)
            @test_nowarn dot(tx, tA, ty)
            @test_nowarn dot(sx, tA, ry)
            @test_nowarn dot(rx, tA, sy)
            @test_nowarn dot(sx, tA, sy)
            @test_throws MissingPrimalError dot(sx, tA, ty)
            @test_throws MissingPrimalError dot(sx, rA, ty)
            @test_nowarn dot(rx, sA, ty)
            @test_nowarn dot(sx, sA, ty)
        end
    end
end

@testset "Eigenvalues" begin
    t1 = idx2tracer([1, 3, 4])
    t2 = idx2tracer([2, 4])
    t3 = idx2tracer([8, 9])
    t4 = idx2tracer([8, 9])
    A = [t1 t2; t3 t4]
    s_out = idx2set([1, 2, 3, 4, 8, 9])
    values, vectors = eigen(A)
    @test size(values) == (2,)
    @test size(vectors) == (2, 2)
    @test all(t -> sameidx(t, s_out), values)
    @test all(t -> sameidx(t, s_out), vectors)
end

@testset "SparseMatrixCSC construction" begin
    t1 = idx2tracer(1)
    t2 = idx2tracer(2)
    t3 = idx2tracer(3)
    SA = sparse([t1 t2; t3 0])
    @test length(SA.nzval) == 3

    res = opnorm(SA, 1)
    @test sameidx(res, [1, 2, 3])
end
