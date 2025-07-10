using SparseConnectivityTracer
using SparseConnectivityTracer: GradientTracer, HessianTracer, Dual
using Test

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("tracers_definitions.jl")

@testset "Jacobian" begin
    P = Float32
    x = rand(P, 3, 2)
    @testset "$GP" for GP in GRADIENT_PATTERNS
        T = GradientTracer{GP}
        D = Dual{P, T}
        @testset "Global" begin
            detector = TracerSparsityDetector(; gradient_tracer_type = T)
            buff = jacobian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == T
        end
        @testset "Local" begin
            detector = TracerLocalSparsityDetector(; gradient_tracer_type = T)
            buff = jacobian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == D
        end
    end
end

@testset "Hessian" begin
    P = Float32
    x = rand(P, 3, 2)
    @testset "$HP" for HP in HESSIAN_PATTERNS
        T = HessianTracer{HP}
        D = Dual{P, T}
        @testset "Global" begin
            detector = TracerSparsityDetector(; hessian_tracer_type = T)
            buff = hessian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == T
        end
        @testset "Local" begin
            detector = TracerLocalSparsityDetector(; hessian_tracer_type = T)
            buff = hessian_buffer(x, detector)
            @test size(buff) == (3, 2)
            @test eltype(buff) == D
        end
    end
end
