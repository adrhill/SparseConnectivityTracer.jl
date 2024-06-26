using SparseConnectivityTracer
using SparseConnectivityTracer: Dual, HessianTracer, MissingPrimalError, trace_input, empty
using ADTypes: hessian_sparsity
using SpecialFunctions: erf, beta
using Test

# Load definitions of CONNECTIVITY_TRACERS, GRADIENT_TRACERS, HESSIAN_TRACERS
include("tracers_definitions.jl")

@testset "Global Hessian" begin
    @testset "Default hessian_pattern" begin
        h = hessian_pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4))
        @test h == [
            0 1 0 0
            1 1 0 0
            0 0 0 0
            0 0 0 1
        ]
    end

    @testset "$T" for T in HESSIAN_TRACERS
        method = TracerSparsityDetector(; hessian_tracer_type=T)

        @test hessian_sparsity(identity, rand(), method) ≈ [0;;]
        @test hessian_sparsity(sqrt, rand(), method) ≈ [1;;]

        @test hessian_sparsity(x -> 1 * x, rand(), method) ≈ [0;;]
        @test hessian_sparsity(x -> x * 1, rand(), method) ≈ [0;;]

        # Code coverage
        @test hessian_sparsity(typemax, 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> round(x, RoundNearestTiesUp), 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> 0, 1, method) ≈ [0;;]

        h = hessian_sparsity(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4), method)
        @test h == [
            0 1 0 0
            1 1 0 0
            0 0 0 0
            0 0 0 1
        ]

        h = hessian_sparsity(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], rand(4), method)
        @test h == [
            0 1 0 0
            1 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        h = hessian_sparsity(x -> (x[1] * x[2]) * (x[3] * x[4]), rand(4), method)
        @test h == [
            0 1 1 1
            1 0 1 1
            1 1 0 1
            1 1 1 0
        ]

        h = hessian_sparsity(x -> (x[1] + x[2]) * (x[3] + x[4]), rand(4), method)
        @test h == [
            0 0 1 1
            0 0 1 1
            1 1 0 0
            1 1 0 0
        ]

        h = hessian_sparsity(x -> (x[1] + x[2] + x[3] + x[4])^2, rand(4), method)
        @test h == [
            1 1 1 1
            1 1 1 1
            1 1 1 1
            1 1 1 1
        ]

        h = hessian_sparsity(x -> 1 / (x[1] + x[2] + x[3] + x[4]), rand(4), method)
        @test h == [
            1 1 1 1
            1 1 1 1
            1 1 1 1
            1 1 1 1
        ]

        h = hessian_sparsity(x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), rand(4), method)
        @test h == [
            0 0 0 0
            0 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        h = hessian_sparsity(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4), method)
        @test h == [
            0 1 0 0
            1 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        h = hessian_sparsity(x -> div(x[1] * x[2], x[3] * x[4]), rand(4), method)
        @test h == [
            0 0 0 0
            0 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        h = hessian_sparsity(x -> sum(sincosd(x)), 1.0, method)
        @test h ≈ [1;;]

        h = hessian_sparsity(x -> sum(diff(x) .^ 3), rand(4), method)
        @test h == [
            1 1 0 0
            1 1 1 0
            0 1 1 1
            0 0 1 1
        ]

        x = rand(5)
        foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
        h = hessian_sparsity(foo, x, method)
        @test h == [
            0 0 0 0 0
            0 0 1 0 0
            0 1 0 0 0
            0 0 0 1 0
            0 0 0 0 0
        ]

        bar(x) = foo(x) + x[2]^x[5]
        h = hessian_sparsity(bar, x, method)
        @test h == [
            0 0 0 0 0
            0 1 1 0 1
            0 1 0 0 0
            0 0 0 1 0
            0 1 0 0 1
        ]

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
            @test_throws MissingPrimalError hessian_sparsity(f, rand(), method)
        end

        # ifelse and comparisons
        if VERSION >= v"1.8"
            h = hessian_sparsity(x -> ifelse(x[1], x[1]^x[2], x[3] * x[4]), rand(4), method)
            @test h == [
                1  1  0  0
                1  1  0  0
                0  0  0  1
                0  0  1  0
            ]

            h = hessian_sparsity(x -> ifelse(x[1], x[1]^x[2], 1.0), rand(4), method)
            @test h == [
                1  1  0  0
                1  1  0  0
                0  0  0  0
                0  0  0  0
            ]

            h = hessian_sparsity(x -> ifelse(x[1], 1.0, x[3] * x[4]), rand(4), method)
            @test h == [
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
        @test hessian_sparsity(f_ampgo07, [1.0], method) ≈ [1;;]

        # Error handling when applying non-dual tracers to "local" functions with control flow
        # TypeError: non-boolean (SparseConnectivityTracer.GradientTracer{BitSet}) used in boolean context
        @test_throws TypeError hessian_sparsity(
            x -> x[1] > x[2] ? x[1]^x[2] : x[3] * x[4], rand(4), method
        )

        # SpecialFunctions
        @test hessian_sparsity(x -> erf(x[1]), rand(2), method) == [
            1 0
            0 0
        ]
        @test hessian_sparsity(x -> beta(x[1], x[2]), rand(3), method) == [
            1 1 0
            1 1 0
            0 0 0
        ]
        yield()
    end
end

@testset "Local Hessian" begin
    @testset "$T" for T in HESSIAN_TRACERS
        method = TracerLocalSparsityDetector(; hessian_tracer_type=T)

        f1(x) = x[1] + x[2] * x[3] + 1 / x[4] + x[2] * max(x[1], x[5])
        h = hessian_sparsity(f1, [1.0 3.0 5.0 1.0 2.0], method)
        @test h == [
            0  0  0  0  0
            0  0  1  0  1
            0  1  0  0  0
            0  0  0  1  0
            0  1  0  0  0
        ]
        h = hessian_sparsity(f1, [4.0 3.0 5.0 1.0 2.0], method)
        @test h == [
            0  1  0  0  0
            1  0  1  0  0
            0  1  0  0  0
            0  0  0  1  0
            0  0  0  0  0
        ]

        f2(x) = ifelse(x[2] < x[3], x[1] * x[2], x[3] * x[4])
        h = hessian_sparsity(f2, [1 2 3 4], method)
        @test h == [
            0  1  0  0
            1  0  0  0
            0  0  0  0
            0  0  0  0
        ]
        h = hessian_sparsity(f2, [1 3 2 4], method)
        @test h == [
            0  0  0  0
            0  0  0  0
            0  0  0  1
            0  0  1  0
        ]

        # Code coverage
        @test hessian_sparsity(typemax, 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test hessian_sparsity(x -> 0, 1, method) ≈ [0;;]

        # Test special cases on empty tracer
        @test hessian_sparsity(x -> zero(x)^(2//3), 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> (2//3)^zero(x), 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> zero(x)^ℯ, 1, method) ≈ [0;;]
        @test hessian_sparsity(x -> ℯ^zero(x), 1, method) ≈ [0;;]
        yield()
    end
end
