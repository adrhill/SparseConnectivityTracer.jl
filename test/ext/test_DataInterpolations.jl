
using SparseConnectivityTracer
using DataInterpolations
using Test

# Categorize Interpolations by type of differentiability
interpolations_z = (ConstantInterpolation,)
interpolations_f = (LinearInterpolation,)
interpolations_s = (QuadraticInterpolation,)

u = [1.0, 2.0, 5.0]
t = [0.0, 1.0, 3.0]

@testset "Jacobian Global" begin
    method = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, method)

    @testset "Non-differentiable" begin
        @testset "$TI" for TI in interpolations_z
            interpolant = TI(u, t)
            @test J(interpolant, 2.0) ≈ [0;;]
        end
    end
    @testset "First-order differentiable" begin
        @testset "$TI" for TI in interpolations_f
            interpolant = TI(u, t)
            @test J(interpolant, 2.0) ≈ [1;;]
        end
    end
    @testset "Second-order differentiable" begin
        @testset "$TI" for TI in interpolations_s
            interpolant = TI(u, t)
            @test J(interpolant, 2.0) ≈ [1;;]
        end
    end
end

# TODO: add tests
# @testset "Jacobian Local" begin
#     method = TracerLocalSparsityDetector()
#     J(f, x) = jacobian_sparsity(f, x, method)
# end

@testset "Global Hessian" begin
    method = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, method)

    @testset "Non-differentiable" begin
        @testset "$TI" for TI in interpolations_z
            interpolant = TI(u, t)
            @test H(interpolant, 2.0) ≈ [0;;]
        end
    end
    @testset "First-order differentiable" begin
        @testset "$TI" for TI in interpolations_f
            interpolant = TI(u, t)
            @test H(interpolant, 2.0) ≈ [0;;]
        end
    end
    @testset "Second-order differentiable" begin
        @testset "$TI" for TI in interpolations_s
            interpolant = TI(u, t)
            @test H(interpolant, 2.0) ≈ [1;;]
        end
    end
end

# TODO: add tests
# @testset "Local Hessian" begin
#     method = TracerLocalSparsityDetector()
#     H(f, x) = hessian_sparsity(f, x, method)
# end
