using SparseConnectivityTracer
using SparseConnectivityTracer:
    ConnectivityTracer, GradientTracer, MissingPrimalError, tracer, trace_input, empty
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using ADTypes: jacobian_sparsity
using LinearAlgebra: det, dot, logdet
using SpecialFunctions: erf, beta
using Test

const FIRST_ORDER_SET_TYPES = (
    BitSet, Set{UInt64}, DuplicateVector{UInt64}, RecursiveSet{UInt64}, SortedVector{UInt64}
)

@testset "Connectivity Global" begin
    @testset "Set type $G" for G in FIRST_ORDER_SET_TYPES
        A = rand(1, 3)
        @test connectivity_pattern(x -> only(A * x), rand(3), G) ≈ [1 1 1]

        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test connectivity_pattern(f, rand(3), G) ≈ [1 0 0; 1 1 0; 0 0 1]
        @test connectivity_pattern(identity, rand(), G) ≈ [1;;]
        @test connectivity_pattern(Returns(1), 1, G) ≈ [0;;]

        # Test ConnectivityTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test connectivity_pattern(g, x, G) ≈ [1 1; 1 1; 1 1]

        # Code coverage
        @test connectivity_pattern(x -> [sincos(x)...], 1, G) ≈ [1; 1]
        @test connectivity_pattern(typemax, 1, G) ≈ [0;;]
        @test connectivity_pattern(x -> x^(2//3), 1, G) ≈ [1;;]
        @test connectivity_pattern(x -> (2//3)^x, 1, G) ≈ [1;;]
        @test connectivity_pattern(x -> x^ℯ, 1, G) ≈ [1;;]
        @test connectivity_pattern(x -> ℯ^x, 1, G) ≈ [1;;]
        @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, G) ≈ [1;;]

        # SpecialFunctions
        @test connectivity_pattern(x -> erf(x), 1, G) == [1;;]
        @test connectivity_pattern(x -> beta(x[1], x[2]), rand(3), G) == [1 1 0]

        ## Error handling when applying non-dual tracers to "local" functions with control flow
        @test_throws MissingPrimalError connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], G
        ) ≈ [1 1 0 0]
    end
end

@testset "Connectivity Local" begin
    @testset "Set type $G" for G in FIRST_ORDER_SET_TYPES
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 2 3 4], G
        ) ≈ [1 1 0 0]
        @test local_connectivity_pattern(
            x -> ifelse(x[2] < x[3], x[1] + x[2], x[3] * x[4]), [1 3 2 4], G
        ) ≈ [0 0 1 1]
    end
end

@testset "Jacobian Global" begin
    @testset "Set type $G" for G in FIRST_ORDER_SET_TYPES
        method = TracerSparsityDetector(G)

        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test jacobian_sparsity(f, rand(3), method) ≈ [1 0 0; 1 1 0; 0 0 1]
        @test jacobian_sparsity(identity, rand(), method) ≈ [1;;]
        @test jacobian_sparsity(Returns(1), 1, method) ≈ [0;;]

        # Test GradientTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test jacobian_sparsity(g, x, method) ≈ [1 1; 0 0; 1 0]

        # Code coverage
        @test jacobian_sparsity(x -> [sincos(x)...], 1, method) ≈ [1; 1]
        @test jacobian_sparsity(typemax, 1, method) ≈ [0;;]
        @test jacobian_sparsity(x -> x^(2//3), 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> (2//3)^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> x^ℯ, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> ℯ^x, 1, method) ≈ [1;;]
        @test jacobian_sparsity(x -> round(x, RoundNearestTiesUp), 1, method) ≈ [0;;]

        # Linear Algebra
        @test jacobian_sparsity(x -> dot(x[1:2], x[4:5]), rand(5), method) == [1 1 0 1 1]

        # SpecialFunctions
        @test jacobian_sparsity(x -> erf(x), 1, method) == [1;;]
        @test jacobian_sparsity(x -> beta(x[1], x[2]), rand(3), method) == [1 1 0]

        ## Error handling when applying non-dual tracers to "local" functions with control flow
        @test_throws MissingPrimalError jacobian_sparsity(
            x -> x[1] > x[2] ? x[3] : x[4], [1.0, 2.0, 3.0, 4.0], method
        ) ≈ [0 0 0 1;]
    end
end

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

        # Linear algebra
        @test jacobian_sparsity(logdet, [1.0 -1.0; 2.0 2.0], method) ≈ [1 1 1 1]  # (#68)
        @test jacobian_sparsity(x -> log(det(x)), [1.0 -1.0; 2.0 2.0], method) ≈ [1 1 1 1]
        @test jacobian_sparsity(x -> dot(x[1:2], x[4:5]), [0, 1, 0, 1, 0], method) ==
            [1 0 0 0 1]
    end
end
