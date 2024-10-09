using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, Dual, MissingPrimalError
using Test

using Random: rand, GLOBAL_RNG
using LinearAlgebra: det, dot, logdet

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("tracers_definitions.jl")

REAL_TYPES = (Float64, Int, Bool, UInt8, Float16, Rational{Int})

# These exists to be able to quickly run tests in the REPL.
# NOTE: J gets overwritten inside the testsets.
detector = TracerSparsityDetector()
J(f, x) = jacobian_sparsity(f, x, detector)

@testset "Jacobian Global" begin
    @testset "$P" for P in GRADIENT_PATTERNS
        T = GradientTracer{P}
        detector = TracerSparsityDetector(; gradient_tracer_type=T)
        J(f, x) = jacobian_sparsity(f, x, detector)

        @testset "Trivial examples" begin
            f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
            @test J(f, rand(3)) == [1 0 0; 1 1 0; 0 0 1]
            @test J(identity, rand()) ≈ [1;;]
            @test J(Returns(1), 1) ≈ [0;;]
        end

        # Test GradientTracer on functions with zero derivatives
        @testset "Zero derivatives" begin
            x = rand(2)
            g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
            @test J(g, x) == [1 1; 0 0; 1 0]
            @test J(!, true) ≈ [0;;]
        end

        # Code coverage
        @testset "Miscellaneous" begin
            @test J(x -> [sincos(x)...], 1) ≈ [1; 1]
            @test J(typemax, 1) ≈ [0;;]
            @test J(x -> x^(2//3), 1) ≈ [1;;]
            @test J(x -> (2//3)^x, 1) ≈ [1;;]
            @test J(x -> x^ℯ, 1) ≈ [1;;]
            @test J(x -> ℯ^x, 1) ≈ [1;;]
            @test J(x -> 0, 1) ≈ [0;;]

            # Test special cases on empty tracer
            @test J(x -> zero(x)^(2//3), 1) ≈ [0;;]
            @test J(x -> (2//3)^zero(x), 1) ≈ [0;;]
            @test J(x -> zero(x)^ℯ, 1) ≈ [0;;]
            @test J(x -> ℯ^zero(x), 1) ≈ [0;;]
        end

        # Conversions
        @testset "Conversion" begin
            @testset "to $T" for T in REAL_TYPES
                @test J(x -> convert(T, x), 1.0) ≈ [1;;]
            end
        end

        @testset "Round" begin
            @test J(round, 1.1) ≈ [0;;]
            @test J(x -> round(Int, x), 1.1) ≈ [0;;]
            @test J(x -> round(Bool, x), 1.1) ≈ [0;;]
            @test J(x -> round(Float16, x), 1.1) ≈ [0;;]
            @test J(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
            @test J(x -> round(x; digits=3, base=2), 1.1) ≈ [0;;]
        end

        @testset "Random" begin
            @test J(x -> rand(typeof(x)), 1) ≈ [0;;]
            @test J(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]
        end

        @testset "LinearAlgebra" begin
            @test J(x -> dot(x[1:2], x[4:5]), rand(5)) == [1 1 0 1 1]
        end

        @testset "MissingPrimalError" begin
            @testset "$f" for f in (
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
                @test_throws MissingPrimalError J(f, rand())
            end
        end

        @testset "ifelse and comparisons" begin
            if VERSION >= v"1.8"
                @test J(x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4]) ==
                    [1 1 1 1]
                @test J(x -> ifelse(x[2] < x[3], x[1] + x[2], 1.0), [1 2 3 4]) == [1 1 0 0]
                @test J(x -> ifelse(x[2] < x[3], 1.0, x[3] * x[4]), [1 2 3 4]) == [0 0 1 1]
            end

            function f_ampgo07(x)
                return (x[1] <= 0) * convert(eltype(x), Inf) +
                       sin(x[1]) +
                       sin(10//3 * x[1]) +
                       log(abs(x[1])) - 84//100 * x[1] + 3
            end
            @test J(f_ampgo07, [1.0]) ≈ [1;;]

            # Error handling when applying non-dual tracers to "local" functions with control flow
            # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
            @test_throws TypeError J(
                x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0]
            ) == [0 0 1 1;]
        end
        yield()
    end
end

@testset "Jacobian Local" begin
    @testset "$P" for P in GRADIENT_PATTERNS
        T = GradientTracer{P}
        detector = TracerLocalSparsityDetector(; gradient_tracer_type=T)
        J(f, x) = jacobian_sparsity(f, x, detector)

        @testset "Trivial examples" begin

            # Multiplication
            @test J(x -> x[1] * x[2], [1.0, 1.0]) == [1 1;]
            @test J(x -> x[1] * x[2], [1.0, 0.0]) == [0 1;]
            @test J(x -> x[1] * x[2], [0.0, 1.0]) == [1 0;]
            @test J(x -> x[1] * x[2], [0.0, 0.0]) == [0 0;]

            # Division
            @test J(x -> x[1] / x[2], [1.0, 1.0]) == [1 1;]
            @test J(x -> x[1] / x[2], [0.0, 0.0]) == [1 0;]

            # Maximum
            @test J(x -> max(x[1], x[2]), [1.0, 2.0]) == [0 1;]
            @test J(x -> max(x[1], x[2]), [2.0, 1.0]) == [1 0;]
            @test J(x -> max(x[1], x[2]), [1.0, 1.0]) == [1 1;]

            # Minimum
            @test J(x -> min(x[1], x[2]), [1.0, 2.0]) == [1 0;]
            @test J(x -> min(x[1], x[2]), [2.0, 1.0]) == [0 1;]
            @test J(x -> min(x[1], x[2]), [1.0, 1.0]) == [1 1;]
        end

        # Comparisons
        @testset "Comparisons" begin
            @test J(x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0]) == [0 0 0 1;]
            @test J(x -> x[1] > x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0]) == [0 0 1 0;]
            @test J(x -> x[1] < x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0]) == [0 0 1 0;]
            @test J(x -> x[1] < x[2] ? x[3] : x[4], [2.0, 1.0, 3.0, 4.0]) == [0 0 0 1;]

            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 2.0]) == [0 1;]
            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [2.0, 1.0]) == [1 0;]
            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 1.0]) == [1 0;]

            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 2.0]) == [0 1;]
            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [2.0, 1.0]) == [1 0;]
            @test J(x -> x[1] >= x[2] ? x[1] : x[2], [1.0, 1.0]) == [1 0;]

            @test J(x -> x[1] <= x[2] ? x[1] : x[2], [1.0, 2.0]) == [1 0;]
            @test J(x -> x[1] <= x[2] ? x[1] : x[2], [2.0, 1.0]) == [0 1;]
            @test J(x -> x[1] <= x[2] ? x[1] : x[2], [1.0, 1.0]) == [1 0;]

            @test J(x -> x[1] == x[2] ? x[1] : x[2], [1.0, 2.0]) == [0 1;]
            @test J(x -> x[1] == x[2] ? x[1] : x[2], [2.0, 1.0]) == [0 1;]
            @test J(x -> x[1] == x[2] ? x[1] : x[2], [1.0, 1.0]) == [1 0;]

            @testset "Comparison with $T" for T in REAL_TYPES
                _1 = oneunit(T)
                @test J(x -> x[1] > _1 ? x[1] : x[2], [0.0, 2.0]) == [0 1;]
                @test J(x -> x[1] > _1 ? x[1] : x[2], [2.0, 0.0]) == [1 0;]
                @test J(x -> x[1] >= _1 ? x[1] : x[2], [0.0, 2.0]) == [0 1;]
                @test J(x -> x[1] >= _1 ? x[1] : x[2], [2.0, 0.0]) == [1 0;]
                @test J(x -> x[1] < _1 ? x[1] : x[2], [0.0, 2.0]) == [1 0;]
                @test J(x -> x[1] < _1 ? x[1] : x[2], [2.0, 0.0]) == [0 1;]
                @test J(x -> isless(x[1], _1) ? x[1] : x[2], [0.0, 2.0]) == [1 0;]
                @test J(x -> isless(x[1], _1) ? x[1] : x[2], [2.0, 0.0]) == [0 1;]
                @test J(x -> x[1] <= _1 ? x[1] : x[2], [0.0, 2.0]) == [1 0;]
                @test J(x -> x[1] <= _1 ? x[1] : x[2], [2.0, 0.0]) == [0 1;]
                @test J(x -> _1 > x[2] ? x[1] : x[2], [0.0, 2.0]) == [0 1;]
                @test J(x -> _1 > x[2] ? x[1] : x[2], [2.0, 0.0]) == [1 0;]
                @test J(x -> _1 >= x[2] ? x[1] : x[2], [0.0, 2.0]) == [0 1;]
                @test J(x -> _1 >= x[2] ? x[1] : x[2], [2.0, 0.0]) == [1 0;]
                @test J(x -> _1 < x[2] ? x[1] : x[2], [0.0, 2.0]) == [1 0;]
                @test J(x -> _1 < x[2] ? x[1] : x[2], [2.0, 0.0]) == [0 1;]
                @test J(x -> _1 <= x[2] ? x[1] : x[2], [0.0, 2.0]) == [1 0;]
                @test J(x -> _1 <= x[2] ? x[1] : x[2], [2.0, 0.0]) == [0 1;]
            end
        end
        # Code coverage
        @testset "Miscellaneous" begin
            @test J(x -> [sincos(x)...], 1) ≈ [1; 1]
            @test J(typemax, 1) ≈ [0;;]
            @test J(x -> x^(2//3), 1) ≈ [1;;]
            @test J(x -> (2//3)^x, 1) ≈ [1;;]
            @test J(x -> x^ℯ, 1) ≈ [1;;]
            @test J(x -> ℯ^x, 1) ≈ [1;;]
            @test J(x -> 0, 1) ≈ [0;;]
        end

        # Conversions
        @testset "Conversion" begin
            @testset "Conversion to $T" for T in REAL_TYPES
                @test J(x -> convert(T, x), 1.0) ≈ [1;;]
            end
        end
        @testset "Round" begin
            @test J(round, 1.1) ≈ [0;;]
            @test J(x -> round(Int, x), 1.1) ≈ [0;;]
            @test J(x -> round(Bool, x), 1.1) ≈ [0;;]
            @test J(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
            @test J(x -> round(x; digits=3, base=2), 1.1) ≈ [0;;]
        end

        @testset "Random" begin
            @test J(x -> rand(typeof(x)), 1) ≈ [0;;]
            @test J(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]
        end

        @testset "LinearAlgebra." begin
            @test J(logdet, [1.0 -1.0; 2.0 2.0]) == [1 1 1 1]  # (#68)
            @test J(x -> log(det(x)), [1.0 -1.0; 2.0 2.0]) == [1 1 1 1]
            @test J(x -> dot(x[1:2], x[4:5]), [0, 1, 0, 1, 0]) == [1 0 0 0 1]
        end
        yield()
    end
end
