using SparseConnectivityTracer
using SparseConnectivityTracer: Dual, HessianTracer, MissingPrimalError
using SparseConnectivityTracer: trace_input, create_tracers, pattern, shared
using Test

using Random: rand, GLOBAL_RNG
using SpecialFunctions: erf, beta
using NNlib: NNlib

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("tracers_definitions.jl")
REAL_TYPES = (Float64, Int, Bool, UInt8, Float16, Rational{Int})

# These exists to be able to quickly run tests in the REPL.
# NOTE: H gets overwritten inside the testsets.
method = TracerSparsityDetector()
H(f, x) = hessian_sparsity(f, x, method)

P = first(HESSIAN_PATTERNS)
T = HessianTracer{P}
D = Dual{Int,T}

@testset "Global Hessian" begin
    @testset "$P" for P in HESSIAN_PATTERNS
        T = HessianTracer{P}
        method = TracerSparsityDetector(; hessian_tracer_type=T)
        H(f, x) = hessian_sparsity(f, x, method)

        @test H(identity, rand()) ≈ [0;;]
        @test H(sqrt, rand()) ≈ [1;;]

        @test H(x -> 1 * x, rand()) ≈ [0;;]
        @test H(x -> x * 1, rand()) ≈ [0;;]

        # Code coverage
        @test H(sign, 1) ≈ [0;;]
        @test H(typemax, 1) ≈ [0;;]
        @test H(x -> x^(2//3), 1) ≈ [1;;]
        @test H(x -> (2//3)^x, 1) ≈ [1;;]
        @test H(x -> x^ℯ, 1) ≈ [1;;]
        @test H(x -> ℯ^x, 1) ≈ [1;;]
        @test H(x -> 0, 1) ≈ [0;;]

        # Conversions
        @testset "Conversion to $T" for T in REAL_TYPES
            @test H(x -> convert(T, x), 1.0) ≈ [0;;]
            @test H(x -> convert(T, x^2), 1.0) ≈ [1;;]
            @test H(x -> convert(T, x)^2, 1.0) ≈ [1;;]
        end

        # Round
        @test H(round, 1.1) ≈ [0;;]
        @test H(x -> round(Int, x), 1.1) ≈ [0;;]
        @test H(x -> round(Bool, x), 1.1) ≈ [0;;]
        @test H(x -> round(Float16, x), 1.1) ≈ [0;;]
        @test H(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
        @test H(x -> round(x; digits=3, base=2), 1.1) ≈ [0;;]

        # Random
        @test H(x -> rand(typeof(x)), 1) ≈ [0;;]
        @test H(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]

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

        h = H(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4))
        if Bool(shared(T))
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
        if Bool(shared(T))
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

        @test H(x -> sum(sincosd(x)), 1.0) ≈ [1;;]

        @test H(x -> sum(diff(x) .^ 3), rand(4)) == [
            1 1 0 0
            1 1 1 0
            0 1 1 1
            0 0 1 1
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

        # Shared Hessian
        function dead_end(x)
            z = x[1] * x[2]
            return x[3] * x[4]
        end
        h = H(dead_end, rand(4))
        if Bool(shared(T))
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
            @test_throws MissingPrimalError H(f, rand())
        end

        # ifelse and comparisons
        if VERSION >= v"1.8"
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
        end

        function f_ampgo07(x)
            return (x[1] <= 0) * convert(eltype(x), Inf) +
                   sin(x[1]) +
                   sin(10//3 * x[1]) +
                   log(abs(x[1])) - 84//100 * x[1] + 3
        end
        @test H(f_ampgo07, [1.0]) ≈ [1;;]

        # Error handling when applying non-dual tracers to "local" functions with control flow
        # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
        @test_throws TypeError H(x -> x[1] > x[2] ? x[1]^x[2] : x[3] * x[4], rand(4))

        # SpecialFunctions
        @test H(x -> erf(x[1]), rand(2)) == [
            1 0
            0 0
        ]
        @test H(x -> beta(x[1], x[2]), rand(3)) == [
            1 1 0
            1 1 0
            0 0 0
        ]
        yield()
    end
end

@testset "Local Hessian" begin
    @testset "$P" for P in HESSIAN_PATTERNS
        T = HessianTracer{P}
        method = TracerLocalSparsityDetector(; hessian_tracer_type=T)
        H(f, x) = hessian_sparsity(f, x, method)

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
        if Bool(shared(T))
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
        if Bool(shared(T))
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

        # Shared Hessian
        function dead_end(x)
            z = x[1] * x[2]
            return x[3] * x[4]
        end
        h = H(dead_end, rand(4))
        if Bool(shared(T))
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

        # Code coverage
        @test H(sign, 1) ≈ [0;;]
        @test H(typemax, 1) ≈ [0;;]
        @test H(x -> x^(2//3), 1) ≈ [1;;]
        @test H(x -> (2//3)^x, 1) ≈ [1;;]
        @test H(x -> x^ℯ, 1) ≈ [1;;]
        @test H(x -> ℯ^x, 1) ≈ [1;;]
        @test H(x -> 0, 1) ≈ [0;;]

        # Conversions
        @testset "Conversion to $T" for T in REAL_TYPES
            @test H(x -> convert(T, x), 1.0) ≈ [0;;]
            @test H(x -> convert(T, x^2), 1.0) ≈ [1;;]
            @test H(x -> convert(T, x)^2, 1.0) ≈ [1;;]
        end

        # Round
        @test H(round, 1.1) ≈ [0;;]
        @test H(x -> round(Int, x), 1.1) ≈ [0;;]
        @test H(x -> round(Bool, x), 1.1) ≈ [0;;]
        @test H(x -> round(x, RoundNearestTiesAway), 1.1) ≈ [0;;]
        @test H(x -> round(x; digits=3, base=2), 1.1) ≈ [0;;]

        # Random
        @test H(x -> rand(typeof(x)), 1) ≈ [0;;]
        @test H(x -> rand(GLOBAL_RNG, typeof(x)), 1) ≈ [0;;]

        # Test special cases on empty tracer
        @test H(x -> zero(x)^(2//3), 1) ≈ [0;;]
        @test H(x -> (2//3)^zero(x), 1) ≈ [0;;]
        @test H(x -> zero(x)^ℯ, 1) ≈ [0;;]
        @test H(x -> ℯ^zero(x), 1) ≈ [0;;]

        # NNlib extension
        @test H(NNlib.relu, -1) ≈ [0;;]
        @test H(NNlib.relu, 1) ≈ [0;;]
        @test H(NNlib.elu, -1) ≈ [1;;]
        @test H(NNlib.elu, 1) ≈ [0;;]
        @test H(NNlib.celu, -1) ≈ [1;;]
        @test H(NNlib.celu, 1) ≈ [0;;]
        @test H(NNlib.selu, -1) ≈ [1;;]
        @test H(NNlib.selu, 1) ≈ [0;;]

        @test H(NNlib.relu6, -1) ≈ [0;;]
        @test H(NNlib.relu6, 1) ≈ [0;;]
        @test H(NNlib.relu6, 7) ≈ [0;;]

        @test H(NNlib.trelu, 0.9) ≈ [0;;]
        @test H(NNlib.trelu, 1.1) ≈ [0;;]

        @test H(NNlib.swish, -5) ≈ [1;;]
        @test H(NNlib.swish, 0) ≈ [1;;]
        @test H(NNlib.swish, 5) ≈ [1;;]

        @test H(NNlib.hardswish, -5) ≈ [0;;]
        @test H(NNlib.hardswish, 0) ≈ [1;;]
        @test H(NNlib.hardswish, 5) ≈ [0;;]

        @test H(NNlib.hardσ, -4) ≈ [0;;]
        @test H(NNlib.hardσ, 0) ≈ [0;;]
        @test H(NNlib.hardσ, 4) ≈ [0;;]

        @test H(NNlib.hardtanh, -2) ≈ [0;;]
        @test H(NNlib.hardtanh, 0) ≈ [0;;]
        @test H(NNlib.hardtanh, 2) ≈ [0;;]

        @test H(NNlib.softshrink, -1) ≈ [0;;]
        @test H(NNlib.softshrink, 0) ≈ [0;;]
        @test H(NNlib.softshrink, 1) ≈ [0;;]
        yield()
    end
end

@testset "Shared IndexSetHessianPattern - same objects" begin
    @testset "$P" for P in HESSIAN_PATTERNS_SHARED
        T = HessianTracer{P}

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

        @test pattern(yt[1]).hessian === pattern(yt[2]).hessian
        @test pattern(yt[1]).hessian === pattern(yt[3]).hessian
        @test_broken pattern(yt[1]).hessian === pattern(yt[4]).hessian
    end
end
