using SparseConnectivityTracer
using SparseConnectivityTracer:
    Dual, HessianTracer, MissingPrimalError, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using ADTypes: hessian_sparsity
using SpecialFunctions: erf, beta
using Test

SECOND_ORDER_SET_TYPE_TUPLES = (
    (BitSet, Set{Tuple{Int,Int}}),  #
    (Set{UInt64}, Set{Tuple{UInt64,UInt64}}),
    (DuplicateVector{UInt64}, DuplicateVector{Tuple{UInt64,UInt64}}),
    # (RecursiveSet{UInt64}, RecursiveSet{Tuple{UInt64,UInt64}}),  # TODO: code union!
    # (SortedVector{UInt64}, SortedVector{Tuple{UInt64,UInt64}}),  # TODO: fix bug
)

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

    @testset "Set type $((G,H))" for (G, H) in SECOND_ORDER_SET_TYPE_TUPLES
        method = TracerSparsityDetector(G, H)

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

        ## Error handling when applying non-dual tracers to "local" functions with control flow
        f2(x) = ifelse(x[2] < x[3], x[1] * x[2], x[3] * x[4])
        @test_throws MissingPrimalError hessian_sparsity(f2, [1 3 2 4], method)
    end
end

@testset "Local Hessian" begin
    @testset "Set type $((G, H))" for (G, H) in SECOND_ORDER_SET_TYPE_TUPLES
        method = TracerLocalSparsityDetector(G, H)

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

        # Putting Duals into Duals is prohibited
        H = empty(HessianTracer{G,H})
        D1 = Dual(1.0, H)
        @test_throws ErrorException D2 = Dual(D1, H)
    end
end
