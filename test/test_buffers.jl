using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, HessianTracer, Dual
using Test

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("tracers_definitions.jl")

@testset "Jacobian" begin
    P = Float32
    x = rand(P, 3, 2)
    @testset "$T" for T in GRADIENT_TRACERS
        D = Dual{P, T}
        @testset "Global" begin
            detector = TracerSparsityDetector(T)
            buff = jacobian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == T
        end
        @testset "Local" begin
            detector = TracerLocalSparsityDetector(T)
            buff = jacobian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == D
        end
    end
end

@testset "Hessian" begin
    P = Float32
    x = rand(P, 3, 2)
    @testset "$T" for T in HESSIAN_TRACERS
        D = Dual{P, T}
        @testset "Global" begin
            detector = TracerSparsityDetector(T)
            buff = hessian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == T
        end
        @testset "Local" begin
            detector = TracerLocalSparsityDetector(T)
            buff = hessian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == D
        end
    end
end
