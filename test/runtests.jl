using SparseConnectivityTracer
using SparseConnectivityTracer: trace_input

using Test
using ReferenceTests
using JuliaFormatter
using Aqua
using JET

using LinearAlgebra
using Random
using NNlib

@testset "SparseConnectivityTracer.jl" begin
    @testset "Code formatting" begin
        @test JuliaFormatter.format(
            SparseConnectivityTracer; verbose=false, overwrite=false
        )
    end
    @testset "Aqua.jl tests" begin
        Aqua.test_all(
            SparseConnectivityTracer;
            ambiguities=false,
            deps_compat=(ignore=[:Random, :SparseArrays],),
        )
    end
    @testset "JET tests" begin
        JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
    end

    @testset "Connectivity" begin
        x = rand(3)
        xt = trace_input(x)

        # Matrix multiplication
        A = rand(1, 3)
        yt = only(A * xt)
        @test sortedinputs(yt) == [1, 2, 3]

        @test connectivity(x -> only(A * x), x) ≈ [1 1 1]

        # Custom functions
        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        yt = f(xt)
        @test sortedinputs(yt[1]) == [1]
        @test sortedinputs(yt[2]) == [1, 2]
        @test sortedinputs(yt[3]) == [3]

        @test connectivity(f, x) ≈ [1 0 0; 1 1 0; 0 0 1]

        @test connectivity(identity, rand()) ≈ [1;;]
        @test connectivity(Returns(1), 1) ≈ [0;;]
    end
    @testset "Real-world tests" begin
        @testset "NNlib" begin
            x = rand(3, 3, 2, 1) # WHCN
            w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
            C = connectivity(x -> NNlib.conv(x, w), x)
            @test_reference "references/connectivity/NNlib/conv.txt" BitMatrix(C)
        end
    end
end
