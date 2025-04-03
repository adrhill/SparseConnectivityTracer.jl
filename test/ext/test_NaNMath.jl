using SparseConnectivityTracer
using NaNMath
using Test

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("../tracers_definitions.jl")

nan_1_to_1 = (
    NaNMath.sqrt,
    NaNMath.sin,
    NaNMath.cos,
    NaNMath.tan,
    NaNMath.asin,
    NaNMath.acos,
    NaNMath.acosh,
    NaNMath.atanh,
    NaNMath.log,
    NaNMath.log2,
    NaNMath.log10,
    NaNMath.log1p,
    NaNMath.lgamma,
)

@testset "Jacobian Global" begin
    detector = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in nan_1_to_1
            @test J(x -> f(x[1]), rand(2)) == [1 0]
        end
    end
    @testset "2-to-1 functions" begin
        @test J(x -> NaNMath.pow(x[1], x[2]), rand(3)) == [1 1 0]
        @test J(x -> NaNMath.max(x[1], x[2]), rand(3)) == [1 1 0]
        @test J(x -> NaNMath.min(x[1], x[2]), rand(3)) == [1 1 0]
    end
end

@testset "Jacobian Local" begin
    detector = TracerLocalSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, detector)

    @testset "2-to-1 functions" begin
        @test J(x -> NaNMath.max(x[1], x[2]), [1.0, 2.0, 0.0]) == [0 1 0]
        @test J(x -> NaNMath.max(x[1], x[2]), [2.0, 1.0, 0.0]) == [1 0 0]
        @test J(x -> NaNMath.min(x[1], x[2]), [1.0, 2.0, 0.0]) == [1 0 0]
        @test J(x -> NaNMath.min(x[1], x[2]), [2.0, 1.0, 0.0]) == [0 1 0]
    end
end

@testset "Hessian Global" begin
    detector = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, detector)

    @testset "1-to-1 functions" begin
        @testset "$f" for f in nan_1_to_1
            @test H(x -> f(x[1]), rand(2)) == [1 0; 0 0]
        end
    end
    @testset "2-to-1 functions" begin
        @test H(x -> NaNMath.pow(x[1], x[2]), rand(3)) == [1 1 0; 1 1 0; 0 0 0]
        @test H(x -> NaNMath.max(x[1], x[2]), rand(3)) == zeros(Bool, 3, 3)
        @test H(x -> NaNMath.min(x[1], x[2]), rand(3)) == zeros(Bool, 3, 3)
    end
end

@testset "Hessian Local" begin
    detector = TracerLocalSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, detector)

    @testset "2-to-1 functions" begin
        @test H(x -> NaNMath.max(x[1], x[2]), [1.0, 2.0, 0.0]) == zeros(Bool, 3, 3)
        @test H(x -> NaNMath.max(x[1], x[2]), [2.0, 1.0, 0.0]) == zeros(Bool, 3, 3)
        @test H(x -> NaNMath.min(x[1], x[2]), [1.0, 2.0, 0.0]) == zeros(Bool, 3, 3)
        @test H(x -> NaNMath.min(x[1], x[2]), [2.0, 1.0, 0.0]) == zeros(Bool, 3, 3)
    end
end
