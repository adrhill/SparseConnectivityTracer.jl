using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, Dual, MissingPrimalError
using Test

using Random: rand, GLOBAL_RNG
using LinearAlgebra: I, det, dot, logdet

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("tracers_definitions.jl")

REAL_TYPES = (Float64, Int, Bool, UInt8, Float16, Rational{Int})

# These exists to be able to quickly run tests in the REPL.
# NOTE: J gets overwritten inside the testsets.
detector = TracerSparsityDetector()
J(f, x) = jacobian_sparsity(f, x, detector)
J(f!, y, x) = jacobian_sparsity(f!, y, x, detector)
T = DEFAULT_GRADIENT_TRACER

@testset "Jacobian Global" begin
    @testset "$T" for T in GRADIENT_TRACERS
        detector = TracerSparsityDetector(T)
        J(f, x) = jacobian_sparsity(f, x, detector)
        J(f!, y, x) = jacobian_sparsity(f!, y, x, detector)

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
            @test J(x -> x^(2 // 3), 1) ≈ [1;;]
            @test J(x -> (2 // 3)^x, 1) ≈ [1;;]
            @test J(x -> x^ℯ, 1) ≈ [1;;]
            @test J(x -> ℯ^x, 1) ≈ [1;;]
            @test J(x -> 0, 1) ≈ [0;;]

            # Test special cases on empty tracer
            @test J(x -> zero(x)^(2 // 3), 1) ≈ [0;;]
            @test J(x -> (2 // 3)^zero(x), 1) ≈ [0;;]
            @test J(x -> zero(x)^ℯ, 1) ≈ [0;;]
            @test J(x -> ℯ^zero(x), 1) ≈ [0;;]
        end

        @testset "In-place functions" begin
            x = rand(5)
            y = similar(x)

            function f!(y, x)
                for i in 1:(length(x) - 1)
                    y[i] = x[i + 1] - x[i]
                end
            end
            @test_nowarn J(f!, y, x)
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
            @test J(x -> round(x; digits = 3, base = 2), 1.1) ≈ [0;;]
        end

        @testset "Three-argument operators" begin
            @test J(x -> clamp(x, 0.0, 1.0), rand()) == [1;;]
            @test J(x -> clamp(x[1], x[2], 1.0), rand(2)) == [1 1]
            @test J(x -> clamp(x[1], 0.0, x[2]), rand(2)) == [1 1]
            @test J(x -> clamp(x[1], x[2], x[3]), rand(3)) == [1 1 1]

            @test J(x -> fma(x, 1.0, 1.0), rand()) == [1;;]
            @test J(x -> fma(1.0, x, 1.0), rand()) == [1;;]
            @test J(x -> fma(1.0, 1.0, x), rand()) == [1;;]
            @test J(x -> fma(x[1], x[2], 1.0), rand(2)) == [1 1]
            @test J(x -> fma(x[1], 1.0, x[2]), rand(2)) == [1 1]
            @test J(x -> fma(1.0, x[1], x[2]), rand(2)) == [1 1]
            @test J(x -> fma(x[1], x[2], x[3]), rand(3)) == [1 1 1]
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
                    isnan,
                    isodd,
                    isone,
                    isreal,
                    iszero,
                )
                @test_throws MissingPrimalError J(f, rand())
            end
        end

        @testset "ifelse and comparisons" begin
            @test J(x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4]) ==
                [1 1 1 1]
            @test J(x -> ifelse(x[2] < x[3], x[1] + x[2], 1.0), [1 2 3 4]) == [1 1 0 0]
            @test J(x -> ifelse(x[2] < x[3], 1.0, x[3] * x[4]), [1 2 3 4]) == [0 0 1 1]

            function f_ampgo07(x)
                return (x[1] <= 0) * convert(eltype(x), Inf) +
                    sin(x[1]) +
                    sin(10 // 3 * x[1]) +
                    log(abs(x[1])) - 84 // 100 * x[1] + 3
            end
            @test J(f_ampgo07, [1.0]) ≈ [1;;]

            # Error handling when applying non-dual tracers to "local" functions with control flow
            # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
            @test_throws TypeError J(
                x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0]
            ) == [0 0 1 1;]
        end

        @testset "Output of type Vector{Any}" begin
            function f(x::AbstractVector)
                n = length(x)
                ret = [] # return type will be Vector{Any}
                for i in 1:(n - 1)
                    append!(
                        ret,
                        abs2(x[i + 1]) - abs2(x[i]) + abs2(x[n - i]) - abs2(x[n - i + 1]),
                    )
                end
                return ret
            end
            x = [
                0.263914
                0.605532
                1.281598
                1.413663
                0.178133
                -1.705427
            ]
            @test J(f, x) == [
                1  1  0  0  1  1
                0  1  1  1  1  0
                0  0  1  1  0  0
                0  1  1  1  1  0
                1  1  0  0  1  1
            ]
        end

        # NOTE: If these tests fail, changes might be breaking on stateful code (see PR #248).
        @testset "Ignore multiplication by zero" begin
            f1(x) = [0 * x[1]]
            @test J(f1, [1.0]) == [1;;]
            f2(x) = [x[1] * 0]
            @test J(f2, [1.0]) == [1;;]
        end

        yield()
    end
end

@testset "Jacobian Local" begin
    @testset "$T" for T in GRADIENT_TRACERS
        detector = TracerLocalSparsityDetector(T)
        J(f, x) = jacobian_sparsity(f, x, detector)
        J(f!, y, x) = jacobian_sparsity(f!, y, x, detector)

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
            @test J(x -> x^(2 // 3), 1) ≈ [1;;]
            @test J(x -> (2 // 3)^x, 1) ≈ [1;;]
            @test J(x -> x^ℯ, 1) ≈ [1;;]
            @test J(x -> ℯ^x, 1) ≈ [1;;]
            @test J(x -> 0, 1) ≈ [0;;]
        end

        @testset "In-place functions" begin
            x = rand(5)
            y = similar(x)

            function f!(y, x)
                for i in 1:(length(x) - 1)
                    y[i] = x[i + 1] - x[i]
                end
            end
            @test_nowarn J(f!, y, x)
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
            @test J(x -> round(x; digits = 3, base = 2), 1.1) ≈ [0;;]
        end

        @testset "Three-argument operators" begin
            @test J(x -> clamp(x, 0.0, 1.0), 0.5) == [1;;]
            @test J(x -> clamp(x, 0.0, 1.0), -0.5) == [0;;]
            @test J(x -> clamp(x[1], x[2], 1.0), [0.5, 0.0]) == [1 0]
            @test J(x -> clamp(x[1], x[2], 1.0), [0.5, 0.6]) == [0 1]
            @test J(x -> clamp(x[1], 0.0, x[2]), [0.5, 1.0]) == [1 0]
            @test J(x -> clamp(x[1], 0.0, x[2]), [0.5, 0.4]) == [0 1]
            @test J(x -> clamp(x[1], x[2], x[3]), [0.5, 0.0, 1.0]) == [1 0 0]
            @test J(x -> clamp(x[1], x[2], x[3]), [0.5, 0.6, 1.0]) == [0 1 0]
            @test J(x -> clamp(x[1], x[2], x[3]), [0.5, 0.0, 0.4]) == [0 0 1]

            @test J(x -> fma(x, 1.0, 1.0), 1.0) == [1;;]
            @test J(x -> fma(1.0, x, 1.0), 1.0) == [1;;]
            @test J(x -> fma(1.0, 1.0, x), 1.0) == [1;;]
            @test J(x -> fma(x, 0.0, 1.0), 1.0) == [0;;]
            @test J(x -> fma(0.0, x, 1.0), 1.0) == [0;;]
            @test J(x -> fma(0.0, 1.0, x), 1.0) == [1;;]
            @test J(x -> fma(x, 1.0, 0.0), 1.0) == [1;;]
            @test J(x -> fma(1.0, x, 0.0), 1.0) == [1;;]
            @test J(x -> fma(1.0, 0.0, x), 1.0) == [1;;]
            @test J(x -> fma(x, 0.0, 0.0), 1.0) == [0;;]
            @test J(x -> fma(0.0, x, 0.0), 1.0) == [0;;]
            @test J(x -> fma(0.0, 0.0, x), 1.0) == [1;;]
            @test J(x -> fma(x[1], x[2], 1.0), [1.0, 1.0]) == [1 1]
            @test J(x -> fma(x[1], 1.0, x[2]), [1.0, 1.0]) == [1 1]
            @test J(x -> fma(1.0, x[1], x[2]), [1.0, 1.0]) == [1 1]
            @test J(x -> fma(x[1], x[2], 1.0), [0.0, 1.0]) == [1 0]
            @test J(x -> fma(x[1], 1.0, x[2]), [0.0, 1.0]) == [1 1]
            @test J(x -> fma(1.0, x[1], x[2]), [0.0, 1.0]) == [1 1]
            @test J(x -> fma(x[1], x[2], 1.0), [1.0, 0.0]) == [0 1]
            @test J(x -> fma(x[1], 1.0, x[2]), [1.0, 0.0]) == [1 1]
            @test J(x -> fma(1.0, x[1], x[2]), [1.0, 0.0]) == [1 1]
            @test J(x -> fma(x[1], x[2], 1.0), [0.0, 0.0]) == [0 0]
            @test J(x -> fma(x[1], 1.0, x[2]), [0.0, 0.0]) == [1 1]
            @test J(x -> fma(1.0, x[1], x[2]), [0.0, 0.0]) == [1 1]
            @test J(x -> fma(x[1], x[2], 0.0), [1.0, 1.0]) == [1 1]
            @test J(x -> fma(x[1], 0.0, x[2]), [1.0, 1.0]) == [0 1]
            @test J(x -> fma(0.0, x[1], x[2]), [1.0, 1.0]) == [0 1]
            @test J(x -> fma(x[1], x[2], 0.0), [0.0, 1.0]) == [1 0]
            @test J(x -> fma(x[1], 0.0, x[2]), [0.0, 1.0]) == [0 1]
            @test J(x -> fma(0.0, x[1], x[2]), [0.0, 1.0]) == [0 1]
            @test J(x -> fma(x[1], x[2], 0.0), [1.0, 0.0]) == [0 1]
            @test J(x -> fma(x[1], 0.0, x[2]), [1.0, 0.0]) == [0 1]
            @test J(x -> fma(0.0, x[1], x[2]), [1.0, 0.0]) == [0 1]
            @test J(x -> fma(x[1], x[2], 0.0), [0.0, 0.0]) == [0 0]
            @test J(x -> fma(x[1], 0.0, x[2]), [0.0, 0.0]) == [0 1]
            @test J(x -> fma(0.0, x[1], x[2]), [0.0, 0.0]) == [0 1]
            @test J(x -> fma(x[1], x[2], x[3]), [0.0, 0.0, 0.0]) == [0 0 1]
            @test J(x -> fma(x[1], x[2], x[3]), [1.0, 0.0, 0.0]) == [0 1 1]
            @test J(x -> fma(x[1], x[2], x[3]), [0.0, 1.0, 0.0]) == [1 0 1]
            @test J(x -> fma(x[1], x[2], x[3]), [1.0, 1.0, 0.0]) == [1 1 1]
            @test J(x -> fma(x[1], x[2], x[3]), [0.0, 0.0, 1.0]) == [0 0 1]
            @test J(x -> fma(x[1], x[2], x[3]), [1.0, 0.0, 1.0]) == [0 1 1]
            @test J(x -> fma(x[1], x[2], x[3]), [0.0, 1.0, 1.0]) == [1 0 1]
            @test J(x -> fma(x[1], x[2], x[3]), [1.0, 1.0, 1.0]) == [1 1 1]
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
        @testset "Output of type Vector{Any}" begin
            function f(x::AbstractVector)
                n = length(x)
                ret = [] # return type will be Vector{Any}
                for i in 1:(n - 1)
                    append!(
                        ret,
                        abs2(x[i + 1]) - abs2(x[i]) + abs2(x[n - i]) - abs2(x[n - i + 1]),
                    )
                end
                return ret
            end
            x = [
                0.263914
                0.605532
                1.281598
                1.413663
                0.178133
                -1.705427
            ]
            @test J(f, x) == [
                1  1  0  0  1  1
                0  1  1  1  1  0
                0  0  1  1  0  0
                0  1  1  1  1  0
                1  1  0  0  1  1
            ]
        end

        yield()
    end
end
