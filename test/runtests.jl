using SparseConnectivityTracer
using Test
using JuliaFormatter
using Aqua
using JET

using LinearAlgebra
using Random

@testset "SparseConnectivityTracer.jl" begin
    @testset "Code formatting" begin
        @test JuliaFormatter.format(
            SparseConnectivityTracer; verbose=false, overwrite=false
        )
    end
    @testset "Aqua.jl tests" begin
        Aqua.test_all(SparseConnectivityTracer)
    end
    @testset "JET tests" begin
        JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
    end

    @testset "Connectivity" begin
        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        @test connectivity(f, rand(3)) == BitMatrix([1 0 0; 1 1 0; 0 0 1])

        @test connectivity(identity, rand()) == BitMatrix([1;;])
    end
end
