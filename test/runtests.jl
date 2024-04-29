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
            deps_compat=(ignore=[:Random, :SparseArrays], check_extras=false),
            persistent_tasks=false,
        )
    end
    @testset "JET tests" begin
        JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
    end
    @testset "Doctests" begin
        Documenter.doctest(SparseConnectivityTracer)
    end
    @testset "Classification of operators by diff'ability" begin
        include("test_differentiability.jl")
    end
    @testset "First order" begin
        x = rand(3)
        xt = trace_input(ConnectivityTracer, x)

        # Matrix multiplication
        A = rand(1, 3)
        yt = only(A * xt)
        @test inputs(yt) == [1, 2, 3]

        @test pattern(x -> only(A * x), ConnectivityTracer, x) ≈ [1 1 1]

        # Custom functions
        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        yt = f(xt)
        @test inputs(yt[1]) == [1]
        @test inputs(yt[2]) == [1, 2]
        @test inputs(yt[3]) == [3]

        @test pattern(f, ConnectivityTracer, x) ≈ [1 0 0; 1 1 0; 0 0 1]
        @test pattern(f, JacobianTracer, x) ≈ [1 0 0; 1 1 0; 0 0 1]

        @test pattern(identity, ConnectivityTracer, rand()) ≈ [1;;]
        @test pattern(identity, JacobianTracer, rand()) ≈ [1;;]
        @test pattern(Returns(1), ConnectivityTracer, 1) ≈ [0;;]
        @test pattern(Returns(1), JacobianTracer, 1) ≈ [0;;]

        # Test JacobianTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test pattern(g, ConnectivityTracer, x) ≈ [1 1; 1 1; 1 1]
        @test pattern(g, JacobianTracer, x) ≈ [1 1; 0 0; 1 0]
    end
    @testset "Second order" begin
        @test pattern(identity, HessianTracer, rand()) ≈ [0;;]
        @test pattern(sqrt, HessianTracer, rand()) ≈ [1;;]

        @test pattern(x -> 1 * x, HessianTracer, rand()) ≈ [0;;]
        @test pattern(x -> x * 1, HessianTracer, rand()) ≈ [0;;]

        x = rand(5)
        f(x) = x[1] + x[2] * x[3] + 1 / x[4]
        H = pattern(f, HessianTracer, x)
        @test H ≈ [
            0 0 0 0 0
            0 0 1 0 0
            0 1 0 0 0
            0 0 0 1 0
            0 0 0 0 0
        ]

        g(x) = x[1] + x[2] * x[3] + 1 / x[4] + x[2]^x[5]
        H = pattern(g, HessianTracer, x)
        @test H ≈ [
            0 0 0 0 0
            0 1 1 0 1
            0 1 0 0 0
            0 0 0 1 0
            0 1 0 0 1
        ]
    end
    @testset "Real-world tests" begin
        @testset "NNlib" begin
            x = rand(3, 3, 2, 1) # WHCN
            w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
            C = pattern(x -> NNlib.conv(x, w), ConnectivityTracer, x)
            @test_reference "references/pattern/connectivity/NNlib/conv.txt" BitMatrix(C)
            J = pattern(x -> NNlib.conv(x, w), JacobianTracer, x)
            @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(J)
            @test C == J
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

            C = pattern(f!, du, ConnectivityTracer, u)
            @test_reference "references/pattern/connectivity/Brusselator.txt" BitMatrix(C)
            J = pattern(f!, du, JacobianTracer, u)
            @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
            @test C == J

            C_ref = Symbolics.jacobian_sparsity(f!, du, u)
            @test C == C_ref
        end
    end
    @testset "ADTypes integration" begin
        include("adtypes.jl")
    end
end
