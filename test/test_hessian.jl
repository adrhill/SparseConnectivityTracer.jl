using SparseConnectivityTracer
using SparseConnectivityTracer: Dual, HessianTracer, MissingPrimalError
using SparseConnectivityTracer: create_tracers, hessian, isshared
using LinearAlgebra: I, dot
using Test
using Random: rand, GLOBAL_RNG

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("tracers_definitions.jl")
REAL_TYPES = (Float64, Int, Bool, UInt8, Float16, Rational{Int})

# These exists to be able to quickly run tests in the REPL.
# NOTE: H gets overwritten inside the testsets.
detector = TracerSparsityDetector()
H(f, x) = hessian_sparsity(f, x, detector)

T = DEFAULT_HESSIAN_TRACER
D = Dual{Int, T}

@testset "Global Hessian" begin
    @testset "$T" for T in HESSIAN_TRACERS
        detector = TracerSparsityDetector(T)
        H(f, x) = hessian_sparsity(f, x, detector)

        @testset "Trivial examples" begin
            @test H(identity, rand()) ≈ [0;;]
            @test H(sqrt, rand()) ≈ [1;;]

            @test H(x -> 1 * x, rand()) ≈ [0;;]
            @test H(x -> x * 1, rand()) ≈ [0;;]
        end

        # Code coverage
        @testset "Miscellaneous" begin
            @test H(sign, 1) ≈ [0;;]
            @test H(typemax, 1) ≈ [0;;]
            @test H(x -> x^(2 // 3), 1) ≈ [1;;]
            @test H(x -> (2 // 3)^x, 1) ≈ [1;;]
            @test H(x -> x^ℯ, 1) ≈ [1;;]
            @test H(x -> ℯ^x, 1) ≈ [1;;]
            @test H(x -> 0, 1) ≈ [0;;]

            @test H(x -> sum(sincosd(x)), 1.0) ≈ [1;;]

            @test H(x -> sum(diff(x) .^ 3), rand(4)) == [
                1 1 0 0
                1 1 1 0
                0 1 1 1
                0 0 1 1
            ]
        end

        # Conversions
        @testset "Conversion" begin
            @testset "to $T" for T in REAL_TYPES
                @test H(x -> convert(T, x), 1.0) ≈ [0;;]
                @test H(x -> convert(T, x^2), 1.0) ≈ [1;;]
                @test H(x -> convert(T, x)^2, 1.0) ≈ [1;;]
            end
        end

        @testset "Round" begin
            @test H(round, 1.1) ≈ [0;;]
            @test H(x -> round(Int, x), 1.1) ≈ [0;;]
            @test H(x -> round(Bool, x), 1.1) ≈ [0;;]
            @test H(x -> round(Float16, x), 1.1) ≈ [0;;]
            @test H(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
            @test H(x -> round(x; digits = 3, base = 2), 1.1) ≈ [0;;]
        end

        @testset "Three-argument operators" begin
            @test H(x -> clamp(x, 0.1, 0.9), rand()) == [0;;]
            @test H(x -> clamp(x[1], x[2], 0.9), rand(2)) == [0 0; 0 0]
            @test H(x -> clamp(x[1], 0.1, x[2]), rand(2)) == [0 0; 0 0]
            @test H(x -> clamp(x[1], x[2], x[3]), rand(3)) == [0 0 0; 0 0 0; 0 0 0]
            @test H(x -> x[1] * clamp(x[1], x[2], x[3]), rand(3)) == [1 1 1; 1 0 0; 1 0 0]
            @test H(x -> x[2] * clamp(x[1], x[2], x[3]), rand(3)) == [0 1 0; 1 1 1; 0 1 0]
            @test H(x -> x[3] * clamp(x[1], x[2], x[3]), rand(3)) == [0 0 1; 0 0 1; 1 1 1]

            @test H(x -> fma(x, 1.0, 1.0), rand()) == [0;;]
            @test H(x -> fma(1.0, x, 1.0), rand()) == [0;;]
            @test H(x -> fma(1.0, 1.0, x), rand()) == [0;;]
            @test H(x -> fma(x[1], x[2], 1.0), rand(2)) == [0 1; 1 0]
            @test H(x -> fma(x[1], 1.0, x[2]), rand(2)) == [0 0; 0 0]
            @test H(x -> fma(x[1], x[2], x[3]), rand(3)) == [0 1 0; 1 0 0; 0 0 0]
        end

        @testset "Random" begin
            @test H(x -> rand(typeof(x)), 1) ≈ [0;;]
            @test H(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]
        end

        @testset "Basic operators" begin
            @test H(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4)) == [
                0 1 0 0
                1 1 0 0
                0 0 0 0
                0 0 0 1
            ]

            @test H(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], rand(4)) == [
                0 1 0 0
                1 0 0 0
                0 0 0 0
                0 0 0 0
            ]

            @test H(x -> (x[1] * x[2]) * (x[3] * x[4]), rand(4)) == [
                0 1 1 1
                1 0 1 1
                1 1 0 1
                1 1 1 0
            ]

            @test H(x -> (x[1] + x[2]) * (x[3] + x[4]), rand(4)) == [
                0 0 1 1
                0 0 1 1
                1 1 0 0
                1 1 0 0
            ]

            @test H(x -> (x[1] + x[2] + x[3] + x[4])^2, rand(4)) == [
                1 1 1 1
                1 1 1 1
                1 1 1 1
                1 1 1 1
            ]

            @test H(x -> 1 / (x[1] + x[2] + x[3] + x[4]), rand(4)) == [
                1 1 1 1
                1 1 1 1
                1 1 1 1
                1 1 1 1
            ]

            @test H(x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), rand(4)) == [
                0 0 0 0
                0 0 0 0
                0 0 0 0
                0 0 0 0
            ]

            x = rand(5)
            foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
            @test H(foo, x) == [
                0 0 0 0 0
                0 0 1 0 0
                0 1 0 0 0
                0 0 0 1 0
                0 0 0 0 0
            ]

            bar(x) = foo(x) + x[2]^x[5]
            @test H(bar, x) == [
                0 0 0 0 0
                0 1 1 0 1
                0 1 0 0 0
                0 0 0 1 0
                0 1 0 0 1
            ]
        end

        @testset "Zero derivatives" begin
            h = H(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4))
            if isshared(T)
                @test h == [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 1
                    0 0 1 0
                ]
            else
                @test h == [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]
            end

            h = H(x -> div(x[1] * x[2], x[3] * x[4]), rand(4))
            if isshared(T)
                @test Matrix(h) == [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 1
                    0 0 1 0
                ]
            else
                @test h == [
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]
            end
        end

        @testset "shared Hessian" begin
            function dead_end(x)
                z = x[1] * x[2]
                return x[3] * x[4]
            end
            h = H(dead_end, rand(4))

            if isshared(T)
                @test h == [
                    0  1  0  0
                    1  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            else
                @test h == [
                    0  0  0  0
                    0  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            end
        end

        # Missing primal errors
        @testset "MissingPrimalError" begin
            @testset "$f" for f in (
                    iseven,
                    isfinite,
                    isinf,
                    isinteger,
                    isnan,
                    isodd,
                    isone,
                    isreal,
                    iszero,
                )
                @test_throws MissingPrimalError H(f, rand())
            end
        end

        @testset "ifelse and comparisons" begin
            @test H(x -> ifelse(x[1], x[1]^x[2], x[3] * x[4]), rand(4)) == [
                1  1  0  0
                1  1  0  0
                0  0  0  1
                0  0  1  0
            ]

            @test H(x -> ifelse(x[1], x[1]^x[2], 1.0), rand(4)) == [
                1  1  0  0
                1  1  0  0
                0  0  0  0
                0  0  0  0
            ]

            @test H(x -> ifelse(x[1], 1.0, x[3] * x[4]), rand(4)) == [
                0  0  0  0
                0  0  0  0
                0  0  0  1
                0  0  1  0
            ]

            function f_ampgo07(x)
                return (x[1] <= 0) * convert(eltype(x), Inf) +
                    sin(x[1]) +
                    sin(10 // 3 * x[1]) +
                    log(abs(x[1])) - 84 // 100 * x[1] + 3
            end
            @test H(f_ampgo07, [1.0]) ≈ [1;;]

            # Error handling when applying non-dual tracers to "local" functions with control flow
            # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
            @test_throws TypeError H(x -> x[1] > x[2] ? x[1]^x[2] : x[3] * x[4], rand(4))
        end

        # NOTE: If these tests fail, changes might be breaking on stateful code (see PR #248).
        @testset "Ignore multiplication by zero" begin
            f1(x) = 0 * x[1]^2
            @test H(f1, [1.0]) == [1;;]
            f2(x) = x[1]^2 * 0
            @test H(f2, [1.0]) == [1;;]
        end

        yield()
    end
end

@testset "Local Hessian" begin
    @testset "$T" for T in HESSIAN_TRACERS
        detector = TracerLocalSparsityDetector(T)
        H(f, x) = hessian_sparsity(f, x, detector)

        @testset "Trivial examples" begin
            f1(x) = x[1] + x[2] * x[3] + 1 / x[4] + x[2] * max(x[1], x[5])
            @test H(f1, [1.0 3.0 5.0 1.0 2.0]) == [
                0  0  0  0  0
                0  0  1  0  1
                0  1  0  0  0
                0  0  0  1  0
                0  1  0  0  0
            ]

            @test H(f1, [4.0 3.0 5.0 1.0 2.0]) == [
                0  1  0  0  0
                1  0  1  0  0
                0  1  0  0  0
                0  0  0  1  0
                0  0  0  0  0
            ]

            f2(x) = ifelse(x[2] < x[3], x[1] * x[2], x[3] * x[4])
            h = H(f2, [1 2 3 4])
            if isshared(T)
                @test h == [
                    0  1  0  0
                    1  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            else
                @test h == [
                    0  1  0  0
                    1  0  0  0
                    0  0  0  0
                    0  0  0  0
                ]
            end

            h = H(f2, [1 3 2 4])
            if isshared(T)
                @test h == [
                    0  1  0  0
                    1  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            else
                @test h == [
                    0  0  0  0
                    0  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            end
        end

        @testset "Shared Hessian" begin
            function dead_end(x)
                z = x[1] * x[2]
                return x[3] * x[4]
            end
            h = H(dead_end, rand(4))
            if isshared(T)
                @test h == [
                    0  1  0  0
                    1  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            else
                @test h == [
                    0  0  0  0
                    0  0  0  0
                    0  0  0  1
                    0  0  1  0
                ]
            end
        end

        @testset "Miscellaneous" begin
            @test H(sign, 1) ≈ [0;;]
            @test H(typemax, 1) ≈ [0;;]
            @test H(x -> x^(2 // 3), 1) ≈ [1;;]
            @test H(x -> (2 // 3)^x, 1) ≈ [1;;]
            @test H(x -> x^ℯ, 1) ≈ [1;;]
            @test H(x -> ℯ^x, 1) ≈ [1;;]
            @test H(x -> 0, 1) ≈ [0;;]

            # Test special cases on empty tracer

            @test H(x -> zero(x)^(2 // 3), 1) ≈ [0;;]
            @test H(x -> (2 // 3)^zero(x), 1) ≈ [0;;]
            @test H(x -> zero(x)^ℯ, 1) ≈ [0;;]
            @test H(x -> ℯ^zero(x), 1) ≈ [0;;]
        end

        @testset "Conversion" begin
            @testset "to $T" for T in REAL_TYPES
                @test H(x -> convert(T, x), 1.0) ≈ [0;;]
                @test H(x -> convert(T, x^2), 1.0) ≈ [1;;]
                @test H(x -> convert(T, x)^2, 1.0) ≈ [1;;]
            end
        end

        @testset "Round" begin
            @test H(round, 1.1) ≈ [0;;]
            @test H(x -> round(Int, x), 1.1) ≈ [0;;]
            @test H(x -> round(Bool, x), 1.1) ≈ [0;;]
            @test H(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
            @test H(x -> round(x; digits = 3, base = 2), 1.1) ≈ [0;;]
        end

        @testset "Three-argument operators" begin
            @test H(x -> x * clamp(x, 0.0, 1.0), 0.5) == [1;;]
            @test H(x -> x * clamp(x, 0.0, 1.0), -0.5) == [0;;]
            @test H(x -> sum(x) * clamp(x[1], x[2], 1.0), [0.5, 0.0]) == [1 1; 1 0]
            @test H(x -> sum(x) * clamp(x[1], x[2], 1.0), [0.5, 0.6]) == [0 1; 1 1]
            @test H(x -> sum(x) * clamp(x[1], 0.0, x[2]), [0.5, 1.0]) == [1 1; 1 0]
            @test H(x -> sum(x) * clamp(x[1], 0.0, x[2]), [0.5, 0.4]) == [0 1; 1 1]
            @test H(x -> sum(x) * clamp(x[1], x[2], x[3]), [0.5, 0.0, 1.0]) ==
                [1 1 1; 1 0 0; 1 0 0]
            @test H(x -> sum(x) * clamp(x[1], x[2], x[3]), [0.5, 0.6, 1.0]) ==
                [0 1 0; 1 1 1; 0 1 0]
            @test H(x -> sum(x) * clamp(x[1], x[2], x[3]), [0.5, 0.0, 0.4]) ==
                [0 0 1; 0 0 1; 1 1 1]
        end

        @testset "Random" begin
            @test H(x -> rand(typeof(x)), 1) ≈ [0;;]
            @test H(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]
        end
        yield()
    end
end

@testset "Shared HessianTracer - shared `hessian` fields" begin
    @testset "$T" for T in HESSIAN_TRACERS_SHARED

        function multi_output_for_shared_test(x::AbstractArray)
            z = ones(eltype(x), size(x))
            y1 = x[1]^2 * z[1]
            y2 = z[2] * x[2]^2
            y3 = x[1] * x[2]
            y4 = z[1] * z[2]  # entirely new tracer
            y = [y1, y2, y3, y4]
            return y
        end

        x = rand(2)
        xt = create_tracers(T, x, eachindex(x))
        yt = multi_output_for_shared_test(xt)

        @test hessian(yt[1]) === hessian(yt[2])
        @test hessian(yt[1]) === hessian(yt[3])
        @test_broken hessian(yt[1]) === hessian(yt[4])
    end
end
