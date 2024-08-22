
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

u = [1.0, 2.0, 5.0]
t = [0.0, 1.0, 3.0]

@testset "Jacobian" begin
    @testset "$M" for M in (TracerSparsityDetector, TracerLocalSparsityDetector)
        method = M()
        J(f, x) = jacobian_sparsity(f, x, method)

        @testset "Non-differentiable" begin
            @testset "$T" for T in interpolations_z
                interp = T(u, t)
                @test J(interp, 2.0) ≈ [0;;]
            end
        end
        @testset "First-order differentiable" begin
            @testset "$T" for T in interpolations_f
                interp = T(u, t)
                @test J(interp, 2.0) ≈ [1;;]
            end
        end
        @testset "Second-order differentiable" begin
            @testset "$T" for T in interpolations_s
                interp = T(u, t)
                @test J(interp, 2.0) ≈ [1;;]
            end
        end
    end
end

@testset "Hessian" begin
    @testset "$M" for M in (TracerSparsityDetector, TracerLocalSparsityDetector)
        method = M()
        H(f, x) = hessian_sparsity(f, x, method)

        @testset "Non-differentiable" begin
            @testset "$T" for T in interpolations_z
                interp = T(u, t)
                @test H(interp, 2.0) ≈ [0;;]
            end
        end
        @testset "First-order differentiable" begin
            @testset "$T" for T in interpolations_f
                interp = T(u, t)
                @test H(interp, 2.0) ≈ [0;;]
            end
        end
        @testset "Second-order differentiable" begin
            @testset "$T" for T in interpolations_s
                interp = T(u, t)
                @test H(interp, 2.0) ≈ [1;;]
            end
        end
    end
end
