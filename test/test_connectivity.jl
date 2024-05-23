using SparseConnectivityTracer
using SparseConnectivityTracer:
    ConnectivityTracer, Dual, MissingPrimalError, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using LinearAlgebra: det, dot, logdet
using SpecialFunctions: erf, beta
using NNlib: NNlib
using Test

const FIRST_ORDER_SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)
NNLIB_ACTIVATIONS_S = (
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
NNLIB_ACTIVATIONS_F = (
    NNlib.hardσ,
    NNlib.hardtanh,
    NNlib.leakyrelu,
    NNlib.relu,
    NNlib.relu6,
    NNlib.softshrink,
    NNlib.trelu,
)
NNLIB_ACTIVATIONS = union(NNLIB_ACTIVATIONS_S, NNLIB_ACTIVATIONS_F)

@testset "Connectivity Global" begin
    @testset "Set type $S" for S in FIRST_ORDER_SET_TYPES
        A = rand(1, 3)
        @test connectivity_pattern(x -> only(A * x), rand(3), S) == [1 1 1]

        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test connectivity_pattern(f, rand(3), S) == [1 0 0; 1 1 0; 0 0 1]
        @test connectivity_pattern(identity, rand(), S) ≈ [1;;]
        @test connectivity_pattern(Returns(1), 1, S) ≈ [0;;]

        # Test ConnectivityTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test connectivity_pattern(g, x, S) == [1 1; 1 1; 1 1]

        # Code coverage
        @test connectivity_pattern(x -> [sincos(x)...], 1, S) ≈ [1; 1]
        @test connectivity_pattern(typemax, 1, S) ≈ [0;;]
        @test connectivity_pattern(x -> x^(2//3), 1, S) ≈ [1;;]
        @test connectivity_pattern(x -> (2//3)^x, 1, S) ≈ [1;;]
        @test connectivity_pattern(x -> x^ℯ, 1, S) ≈ [1;;]
        @test connectivity_pattern(x -> ℯ^x, 1, S) ≈ [1;;]
        @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, S) ≈ [1;;]
        @test connectivity_pattern(x -> 0, 1, S) ≈ [0;;]

        # SpecialFunctions extension
        @test connectivity_pattern(x -> erf(x[1]), rand(2), S) == [1 0]
        @test connectivity_pattern(x -> beta(x[1], x[2]), rand(3), S) == [1 1 0]

        # NNlib extension
        for f in NNLIB_ACTIVATIONS
            @test connectivity_pattern(f, 1, S) ≈ [1;;]
        end

        # Error handling when applying non-dual tracers to "local" functions with control flow
        @test_throws MissingPrimalError connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], S
        ) == [1 1 0 0]
    end
end

@testset "Connectivity Local" begin
    @testset "Set type $S" for S in FIRST_ORDER_SET_TYPES
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], S
        ) == [1 1 0 0]
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 3 2 4], S
        ) == [0 0 1 1]
        @test local_connectivity_pattern(x -> 0, 1, S) ≈ [0;;]

        # Putting Duals into Duals is prohibited
        C = empty(ConnectivityTracer{S})
        D1 = Dual(1.0, C)
        @test_throws ErrorException D2 = Dual(D1, C)
    end
end
