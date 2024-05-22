using SparseConnectivityTracer
using SparseConnectivityTracer:
    GradientTracer, MissingPrimalError, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using ADTypes: jacobian_sparsity
using LinearAlgebra: det, dot, logdet
using SpecialFunctions: erf, beta
using NNlib: NNlib
using Test

const FIRST_ORDER_SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)
const NNLIB_ACTIVATIONS_S = (
    NNlib.σ,
    NNlib.celu,
    NNlib.elu,
    NNlib.gelu,
    NNlib.hardswish,
    NNlib.lisht,
    NNlib.logσ,
    NNlib.logcosh,
    NNlib.mish,
    NNlib.selu,
    NNlib.softplus,
    NNlib.softsign,
    NNlib.swish,
    NNlib.sigmoid_fast,
    NNlib.tanhshrink,
    NNlib.tanh_fast,
)
const NNLIB_ACTIVATIONS_F = (
    NNlib.hardσ,
    NNlib.hardtanh,
    NNlib.leakyrelu,
    NNlib.relu,
    NNlib.relu6,
    NNlib.softshrink,
    NNlib.trelu,
)
const NNLIB_ACTIVATIONS = union(NNLIB_ACTIVATIONS_S, NNLIB_ACTIVATIONS_F)

@testset "Jacobian Local" verbose = true begin
    @testset "Set type $G" for G in FIRST_ORDER_SET_TYPES
        method = TracerLocalSparsityDetector(G)

        # Multiplication
        @test jacobian_sparsity(x -> x[1] * x[2], [1.0, 1.0], method) ≈ [1 1;]
        @test jacobian_sparsity(x -> x[1] * x[2], [1.0, 0.0], method) ≈ [0 1;]
        @test jacobian_sparsity(x -> x[1] * x[2], [0.0, 1.0], method) ≈ [1 0;]
        @test jacobian_sparsity(x -> x[1] * x[2], [0.0, 0.0], method) ≈ [0 0;]

        # Division
        @test jacobian_sparsity(x -> x[1] / x[2], [1.0, 1.0], method) ≈ [1 1;]
        @test jacobian_sparsity(x -> x[1] / x[2], [0.0, 0.0], method) ≈ [1 0;]

        # Maximum
        @test jacobian_sparsity(x -> max(x[1], x[2]), [1.0, 2.0], method) ≈ [0 1;]
        @test jacobian_sparsity(x -> max(x[1], x[2]), [2.0, 1.0], method) ≈ [1 0;]
        @test jacobian_sparsity(x -> max(x[1], x[2]), [1.0, 1.0], method) ≈ [1 1;]

        # Minimum
        @test jacobian_sparsity(x -> min(x[1], x[2]), [1.0, 2.0], method) ≈ [1 0;]
        @test jacobian_sparsity(x -> min(x[1], x[2]), [2.0, 1.0], method) ≈ [0 1;]
        @test jacobian_sparsity(x -> min(x[1], x[2]), [1.0, 1.0], method) ≈ [1 1;]

        # Comparisons
        @test jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) ≈ [0 0 0 1;]
        @test jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0], method
        ) ≈ [0 0 1 0;]
        @test jacobian_sparsity(
            x -> x[1] < x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) ≈ [0 0 1 0;]
        @test jacobian_sparsity(
            x -> x[1] < x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0], method
        ) ≈ [0 0 0 1;]

        # Code coverage
        @test jacobian_sparsity(x -> [sincos(x)...], 1, method) ≈ [1; 1]
        @test jacobian_sparsity(typemax, 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> round(x, RoundNearestTiesUp), 1, method) ≈ [0;;]

        # Linear algebra
        @test jacobian_sparsity(logdet, [1.0 -1.0; 2.0 2.0], method) ≈ [1 1 1 1]  # (#68)
        @test jacobian_sparsity(x -> log(det(x)), [1.0 -1.0; 2.0 2.0], method) ≈ [1 1 1 1]
        @test jacobian_sparsity(x -> dot(x[1:2], x[4:5]), [0, 1, 0, 1, 0], method) ==
            [1 0 0 0 1]

        # NNlib extension
        @test jacobian_sparsity(NNlib.relu, -1, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.relu, 1, method) ≈ [1;;]

        @test jacobian_sparsity(NNlib.relu6, -1, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.relu6, 1, method) ≈ [1;;]
        @test jacobian_sparsity(NNlib.relu6, 7, method) ≈ [0;;]

        @test jacobian_sparsity(NNlib.trelu, 0.9, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.trelu, 1.1, method) ≈ [1;;]

        @test jacobian_sparsity(NNlib.hardσ, -4, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.hardσ, 0, method) ≈ [1;;]
        @test jacobian_sparsity(NNlib.hardσ, 4, method) ≈ [0;;]

        @test jacobian_sparsity(NNlib.hardtanh, -2, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.hardtanh, 0, method) ≈ [1;;]
        @test jacobian_sparsity(NNlib.hardtanh, 2, method) ≈ [0;;]

        @test jacobian_sparsity(NNlib.softshrink, -1, method) ≈ [1;;]
        @test jacobian_sparsity(NNlib.softshrink, 0, method) ≈ [0;;]
        @test jacobian_sparsity(NNlib.softshrink, 1, method) ≈ [1;;]
    end
end