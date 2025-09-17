using SparseConnectivityTracer
using ChainRulesCore: ignore_derivatives
using LinearAlgebra
using Test

@testset "$detector" for detector in (TracerSparsityDetector(), TracerLocalSparsityDetector())
    @testset "Jacobian Scalar" begin
        J = jacobian_sparsity(ignore_derivatives, 1, detector)
        @test J == [0;;]
    end
    @testset "Jacobian Array" begin
        J = jacobian_sparsity(ignore_derivatives, [1, 2], detector)
        @test J == [0 0; 0 0]
    end
    @testset "Hessian Scalar" begin
        H = hessian_sparsity(x -> ignore_derivatives(x^2), 1, detector)
        @test H == [0;;]
        H = hessian_sparsity(x -> ignore_derivatives(x)^2, 1, detector)
        @test H == [0;;]
    end
    @testset "Hessian Array" begin
        H = hessian_sparsity(x -> logdet(ignore_derivatives(x)), [1.0 0.0; 0.0 1.0], detector)
        @test H == zeros(Bool, 4, 4)
    end

end
