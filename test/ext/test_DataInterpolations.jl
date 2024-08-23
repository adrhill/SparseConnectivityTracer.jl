
using SparseConnectivityTracer
using DataInterpolations
using DataInterpolations: AbstractInterpolation
using Test

# Categorize Interpolations by type of differentiability
interpolations_z = (ConstantInterpolation,)
interpolations_f = (LinearInterpolation,)
interpolations_s = (
    QuadraticInterpolation,
    LagrangeInterpolation,
    AkimaInterpolation,
    QuadraticSpline,
    CubicSpline,
)


t = [0.0, 1.0, 2.5, 4.0];
n = length(t)

uv = rand(n); # vector
um = rand(2, n); # matrix

@testset "$M" for M in (TracerSparsityDetector, TracerLocalSparsityDetector)
    method = M()
    J(f, x) = jacobian_sparsity(f, x, method)
    H(f, x) = hessian_sparsity(f, x, method)

    @testset "Zero first derivative, zero second derivative" begin
        @testset "$T" for T in interpolations_z
            interp_v = T(uv, t)
            interp_m = T(um, t)

            @testset "Jacobian" begin
                @test J(interp_v, 2.0) ≈ [0;;]
                @test J(interp_m, 2.0) == zeros(n)
            end
            @testset "Hessian" begin
                @test H(interp_v, 2.0) ≈ [0;;]
                @test H(x -> sum(interp_m), 2.0) == zeros(n)
            end
        end
    end
    @testset "Non-zero first derivative, zero second derivative" begin
        @testset "$T" for T in interpolations_f
            interp_v = T(uv, t)
            interp_m = T(um, t)

            @testset "Jacobian" begin
                @test J(interp_v, 2.0) ≈ [1;;]
                @test J(interp_m, 2.0) == ones(n)
            end
            @testset "Hessian" begin
                @test H(interp_v, 2.0) ≈ [0;;]
                @test H(x -> sum(interp_m), 2.0) == zeros(n)
            end
        end
    end
    @testset "Non-zero first derivative, non-zero second derivative" begin
        @testset "$T" for T in interpolations_s
            interp_v = T(uv, t)
            interp_m = T(um, t)

            @testset "Jacobian" begin
                @test J(interp_v, 2.0) ≈ [1;;]
                @test J(interp_m, 2.0) == ones(n)
            end
            @testset "Hessian" begin
                @test H(interp_v, 2.0) ≈ [1;;]
                @test H(x -> sum(interp_m), 2.0) == ones(n)
            end
        end
    end
end
