using SparseConnectivityTracer
using SparseConnectivityTracer: ConnectivityTracer, Dual, MissingPrimalError, trace_input
using LinearAlgebra: det, dot, logdet
using SpecialFunctions: erf, beta
using NNlib: NNlib
using Test

# Load definitions of CONNECTIVITY_TRACERS, GRADIENT_TRACERS, HESSIAN_TRACERS
include("tracers_definitions.jl")

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
    @testset "$T" for T in CONNECTIVITY_TRACERS
        A = rand(1, 3)
        @test connectivity_pattern(x -> only(A * x), rand(3), T) == [1 1 1]

        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test connectivity_pattern(f, rand(3), T) == [1 0 0; 1 1 0; 0 0 1]
        @test connectivity_pattern(identity, rand(), T) ≈ [1;;]
        @test connectivity_pattern(Returns(1), 1, T) ≈ [0;;]

        # Test ConnectivityTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test connectivity_pattern(g, x, T) == [1 1; 1 1; 1 1]

        # Code coverage
        @test connectivity_pattern(x -> [sincos(x)...], 1, T) ≈ [1; 1]
        @test connectivity_pattern(typemax, 1, T) ≈ [0;;]
        @test connectivity_pattern(x -> x^(2//3), 1, T) ≈ [1;;]
        @test connectivity_pattern(x -> (2//3)^x, 1, T) ≈ [1;;]
        @test connectivity_pattern(x -> x^ℯ, 1, T) ≈ [1;;]
        @test connectivity_pattern(x -> ℯ^x, 1, T) ≈ [1;;]
        @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, T) ≈ [1;;]
        @test connectivity_pattern(x -> 0, 1, T) ≈ [0;;]

        # SpecialFunctions extension
        @test connectivity_pattern(x -> erf(x[1]), rand(2), T) == [1 0]
        @test connectivity_pattern(x -> beta(x[1], x[2]), rand(3), T) == [1 1 0]

        # NNlib extension
        for f in NNLIB_ACTIVATIONS
            @test connectivity_pattern(f, 1, T) ≈ [1;;]
        end

        # Missing primal errors
        @testset "MissingPrimalError on $f" for f in (
            iseven,
            isfinite,
            isinf,
            isinteger,
            ismissing,
            isnan,
            isnothing,
            isodd,
            isone,
            isreal,
            iszero,
        )
            @test_throws MissingPrimalError connectivity_pattern(f, rand(), T)
        end

        # ifelse and comparisons
        if VERSION >= v"1.8"
            @test connectivity_pattern(
                x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], T
            ) == [1 1 1 1]

            @test connectivity_pattern(
                x -> ifelse(x[2] < x[3], x[1] + x[2], 1.0), [1 2 3 4], T
            ) == [1 1 0 0]

            @test connectivity_pattern(
                x -> ifelse(x[2] < x[3], 1.0, x[3] * x[4]), [1 2 3 4], T
            ) == [0 0 1 1]
        end

        function f_ampgo07(x)
            return (x[1] <= 0) * convert(eltype(x), Inf) +
                   sin(x[1]) +
                   sin(10//3 * x[1]) +
                   log(abs(x[1])) - 84//100 * x[1] + 3
        end
        @test connectivity_pattern(f_ampgo07, [1.0], T) ≈ [1;;]

        # Error handling when applying non-dual tracers to "local" functions with control flow
        # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
        @test_throws TypeError connectivity_pattern(
            x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], T
        ) == [0 0 1 1;]
    end
end

@testset "Connectivity Local" begin
    @testset "$T" for T in CONNECTIVITY_TRACERS
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], T
        ) == [1 1 0 0]
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 3 2 4], T
        ) == [0 0 1 1]
        @test local_connectivity_pattern(x -> 0, 1, T) ≈ [0;;]
    end
end
