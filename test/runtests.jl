using SparseConnectivityTracer
using SparseConnectivityTracer: trace_input

using Test
using ReferenceTests
using JuliaFormatter
using Aqua
using JET
using Documenter

using LinearAlgebra
using Random
using Symbolics: Symbolics
using NNlib

DocMeta.setdocmeta!(
    SparseConnectivityTracer,
    :DocTestSetup,
    :(using SparseConnectivityTracer);
    recursive=true,
)

@testset verbose = true "SparseConnectivityTracer.jl" begin
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
        @testset "Brusselator" begin
            include("brusselator.jl")
            N = 6
            dims = (N, N, 2)
            A = 1.0
            B = 1.0
            alpha = 1.0
            xyd = fill(1.0, N)
            dx = 1.0
            p = (A, B, alpha, xyd, dx, N)

            u = rand(dims...)
            du = similar(u)
            f!(du, u) = brusselator_2d_loop(du, u, p, nothing)

            C = connectivity(f!, du, u)
            @test_reference "references/connectivity/Brusselator.txt" BitMatrix(C)

            C_ref = Symbolics.jacobian_sparsity(f!, du, u)
            @test C == C_ref
        end
    end
    @testset "Doctests" begin
        Documenter.doctest(SparseConnectivityTracer)
    end
end
