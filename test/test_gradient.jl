using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, Dual, MissingPrimalError, trace_input
using ADTypes: jacobian_sparsity
using LinearAlgebra: det, dot, logdet
using SpecialFunctions: erf, beta
using NNlib: NNlib
using Test

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
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

@testset "Jacobian Global" begin
    @testset "$P" for P in GRADIENT_PATTERNS
        T = GradientTracer{P}
        method = TracerSparsityDetector(; gradient_tracer_type=T)

        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test jacobian_sparsity(f, rand(3), method) == [1 0 0; 1 1 0; 0 0 1]
        @test jacobian_sparsity(identity, rand(), method) ≈ [1;;]
        @test jacobian_sparsity(Returns(1), 1, method) ≈ [0;;]

        # Test GradientTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test jacobian_sparsity(g, x, method) == [1 1; 0 0; 1 0]
        @test jacobian_sparsity(!, true, method) ≈ [0;;]

        # Code coverage
        @test jacobian_sparsity(x -> [sincos(x)...], 1, method) ≈ [1; 1]
        @test jacobian_sparsity(typemax, 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> round(x, RoundNearestTiesUp), 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> 0, 1, method) ≈ [0;;]

        # Test special cases on empty tracer
        @test jacobian_sparsity(x -> zero(x)^(2//3), 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> (2//3)^zero(x), 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> zero(x)^ℯ, 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> ℯ^zero(x), 1, method) ≈ [0;;]

        # Linear Algebra
        @test jacobian_sparsity(x -> dot(x[1:2], x[4:5]), rand(5), method) == [1 1 0 1 1]

        # SpecialFunctions extension
        @test jacobian_sparsity(x -> erf(x[1]), rand(2), method) == [1 0]
        @test jacobian_sparsity(x -> beta(x[1], x[2]), rand(3), method) == [1 1 0]

        # Missing primal errors
        @testset "MissingPrimalError on $f" for f in (
            iseven,
            isfinite,
            isinf,
            isinteger,
            isless,
            ismissing,
            isnan,
            isnothing,
            isodd,
            isone,
            isreal,
            iszero,
        )
            @test_throws MissingPrimalError jacobian_sparsity(f, rand(), method)
        end

        # NNlib extension
        for f in NNLIB_ACTIVATIONS
            @test jacobian_sparsity(f, 1, method) ≈ [1;;]
        end

        # ifelse and comparisons
        if VERSION >= v"1.8"
            @test jacobian_sparsity(
                x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], method
            ) == [1 1 1 1]

            @test jacobian_sparsity(
                x -> ifelse(x[2] < x[3], x[1] + x[2], 1.0), [1 2 3 4], method
            ) == [1 1 0 0]

            @test jacobian_sparsity(
                x -> ifelse(x[2] < x[3], 1.0, x[3] * x[4]), [1 2 3 4], method
            ) == [0 0 1 1]
        end

        function f_ampgo07(x)
            return (x[1] <= 0) * convert(eltype(x), Inf) +
                   sin(x[1]) +
                   sin(10//3 * x[1]) +
                   log(abs(x[1])) - 84//100 * x[1] + 3
        end
        @test jacobian_sparsity(f_ampgo07, [1.0], method) ≈ [1;;]

        ## Error handling when applying non-dual tracers to "local" functions with control flow
        # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
        @test_throws TypeError jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) == [0 0 1 1;]
        yield()
    end
end

@testset "Jacobian Local" begin
    @testset "$P" for P in GRADIENT_PATTERNS
        T = GradientTracer{P}
        method = TracerLocalSparsityDetector(; gradient_tracer_type=T)

        # Multiplication
        @test jacobian_sparsity(x -> x[1] * x[2], [1.0, 1.0], method) == [1 1;]
        @test jacobian_sparsity(x -> x[1] * x[2], [1.0, 0.0], method) == [0 1;]
        @test jacobian_sparsity(x -> x[1] * x[2], [0.0, 1.0], method) == [1 0;]
        @test jacobian_sparsity(x -> x[1] * x[2], [0.0, 0.0], method) == [0 0;]

        # Division
        @test jacobian_sparsity(x -> x[1] / x[2], [1.0, 1.0], method) == [1 1;]
        @test jacobian_sparsity(x -> x[1] / x[2], [0.0, 0.0], method) == [1 0;]

        # Maximum
        @test jacobian_sparsity(x -> max(x[1], x[2]), [1.0, 2.0], method) == [0 1;]
        @test jacobian_sparsity(x -> max(x[1], x[2]), [2.0, 1.0], method) == [1 0;]
        @test jacobian_sparsity(x -> max(x[1], x[2]), [1.0, 1.0], method) == [1 1;]

        # Minimum
        @test jacobian_sparsity(x -> min(x[1], x[2]), [1.0, 2.0], method) == [1 0;]
        @test jacobian_sparsity(x -> min(x[1], x[2]), [2.0, 1.0], method) == [0 1;]
        @test jacobian_sparsity(x -> min(x[1], x[2]), [1.0, 1.0], method) == [1 1;]

        # Comparisons
        @test jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) == [0 0 0 1;]
        @test jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0], method
        ) == [0 0 1 0;]
        @test jacobian_sparsity(
            x -> x[1] < x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) == [0 0 1 0;]
        @test jacobian_sparsity(
            x -> x[1] < x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0], method
        ) == [0 0 0 1;]

        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 2.0], method) ==
            [0 1;]
        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [2.0, 1.0], method) ==
            [1 0;]
        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 1.0], method) ==
            [1 0;]

        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 2.0], method) ==
            [0 1;]
        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [2.0, 1.0], method) ==
            [1 0;]
        @test jacobian_sparsity(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 1.0], method) ==
            [1 0;]

        @test jacobian_sparsity(x -> x[1] <= x[2] ? x[1] : x[2], [1.0, 2.0], method) ==
            [1 0;]
        @test jacobian_sparsity(x -> x[1] <= x[2] ? x[1] : x[2], [2.0, 1.0], method) ==
            [0 1;]
        @test jacobian_sparsity(x -> x[1] <= x[2] ? x[1] : x[2], [1.0, 1.0], method) ==
            [1 0;]

        @test jacobian_sparsity(x -> x[1] == x[2] ? x[1] : x[2], [1.0, 2.0], method) ==
            [0 1;]
        @test jacobian_sparsity(x -> x[1] == x[2] ? x[1] : x[2], [2.0, 1.0], method) ==
            [0 1;]
        @test jacobian_sparsity(x -> x[1] == x[2] ? x[1] : x[2], [1.0, 1.0], method) ==
            [1 0;]

        @test jacobian_sparsity(x -> x[1] > 1 ? x[1] : x[2], [0.0, 2.0], method) == [0 1;]
        @test jacobian_sparsity(x -> x[1] > 1 ? x[1] : x[2], [2.0, 0.0], method) == [1 0;]
        @test jacobian_sparsity(x -> x[1] >= 1 ? x[1] : x[2], [0.0, 2.0], method) == [0 1;]
        @test jacobian_sparsity(x -> x[1] >= 1 ? x[1] : x[2], [2.0, 0.0], method) == [1 0;]
        @test jacobian_sparsity(x -> x[1] < 1 ? x[1] : x[2], [0.0, 2.0], method) == [1 0;]
        @test jacobian_sparsity(x -> x[1] < 1 ? x[1] : x[2], [2.0, 0.0], method) == [0 1;]
        @test jacobian_sparsity(x -> x[1] <= 1 ? x[1] : x[2], [0.0, 2.0], method) == [1 0;]
        @test jacobian_sparsity(x -> x[1] <= 1 ? x[1] : x[2], [2.0, 0.0], method) == [0 1;]
        @test jacobian_sparsity(x -> 1 > x[2] ? x[1] : x[2], [0.0, 2.0], method) == [0 1;]
        @test jacobian_sparsity(x -> 1 > x[2] ? x[1] : x[2], [2.0, 0.0], method) == [1 0;]
        @test jacobian_sparsity(x -> 1 >= x[2] ? x[1] : x[2], [0.0, 2.0], method) == [0 1;]
        @test jacobian_sparsity(x -> 1 >= x[2] ? x[1] : x[2], [2.0, 0.0], method) == [1 0;]
        @test jacobian_sparsity(x -> 1 < x[2] ? x[1] : x[2], [0.0, 2.0], method) == [1 0;]
        @test jacobian_sparsity(x -> 1 < x[2] ? x[1] : x[2], [2.0, 0.0], method) == [0 1;]
        @test jacobian_sparsity(x -> 1 <= x[2] ? x[1] : x[2], [0.0, 2.0], method) == [1 0;]
        @test jacobian_sparsity(x -> 1 <= x[2] ? x[1] : x[2], [2.0, 0.0], method) == [0 1;]

        # Code coverage
        @test jacobian_sparsity(x -> [sincos(x)...], 1, method) ≈ [1; 1]
        @test jacobian_sparsity(typemax, 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> round(x, RoundNearestTiesUp), 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> 0, 1, method) ≈ [0;;]

        # Linear algebra
        @test jacobian_sparsity(logdet, [1.0 -1.0; 2.0 2.0], method) == [1 1 1 1]  # (#68)
        @test jacobian_sparsity(x -> log(det(x)), [1.0 -1.0; 2.0 2.0], method) == [1 1 1 1]
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
        yield()
    end
end
