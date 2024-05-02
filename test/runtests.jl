using SparseConnectivityTracer
using SparseConnectivityTracer: trace_input, empty

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
        xt = trace_input(ConnectivityTracer{BitSet}, x)

        # Matrix multiplication
        A = rand(1, 3)
        yt = only(A * xt)
        @test inputs(yt) == [1, 2, 3]

        @test pattern(x -> only(A * x), ConnectivityTracer{BitSet}, x) ≈ [1 1 1]

        # Custom functions
        f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
        yt = f(xt)
        @test inputs(yt[1]) == [1]
        @test inputs(yt[2]) == [1, 2]
        @test inputs(yt[3]) == [3]

        @test pattern(f, ConnectivityTracer{BitSet}, x) ≈ [1 0 0; 1 1 0; 0 0 1]
        @test pattern(f, JacobianTracer{BitSet}, x) ≈ [1 0 0; 1 1 0; 0 0 1]

        @test pattern(identity, ConnectivityTracer{BitSet}, rand()) ≈ [1;;]
        @test pattern(identity, JacobianTracer{BitSet}, rand()) ≈ [1;;]
        @test pattern(Returns(1), ConnectivityTracer{BitSet}, 1) ≈ [0;;]
        @test pattern(Returns(1), JacobianTracer{BitSet}, 1) ≈ [0;;]

        # Test JacobianTracer on functions with zero derivatives
        x = rand(2)
        g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
        @test pattern(g, ConnectivityTracer{BitSet}, x) ≈ [1 1; 1 1; 1 1]
        @test pattern(g, JacobianTracer{BitSet}, x) ≈ [1 1; 0 0; 1 0]

        # Code coverage
        @test pattern(x -> [sincos(x)...], ConnectivityTracer{BitSet}, 1) ≈ [1; 1]
        @test pattern(x -> [sincos(x)...], JacobianTracer{BitSet}, 1) ≈ [1; 1]
        @test pattern(typemax, ConnectivityTracer{BitSet}, 1) ≈ [0;;]
        @test pattern(typemax, JacobianTracer{BitSet}, 1) ≈ [0;;]
        @test pattern(x -> x^(2//3), ConnectivityTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> x^(2//3), JacobianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> (2//3)^x, ConnectivityTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> (2//3)^x, JacobianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> x^ℯ, ConnectivityTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> x^ℯ, JacobianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> ℯ^x, ConnectivityTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> ℯ^x, JacobianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> round(x, RoundNearestTiesUp), ConnectivityTracer{BitSet}, 1) ≈
            [1;;]
        @test pattern(x -> round(x, RoundNearestTiesUp), JacobianTracer{BitSet}, 1) ≈ [0;;]

        @test rand(ConnectivityTracer{BitSet}) == empty(ConnectivityTracer{BitSet})
        @test rand(JacobianTracer{BitSet}) == empty(JacobianTracer{BitSet})

        t = tracer(ConnectivityTracer{BitSet}, 2)
        @test ConnectivityTracer(t) == t
        @test empty(t) == empty(ConnectivityTracer{BitSet})
        @test ConnectivityTracer{BitSet}(1) == empty(ConnectivityTracer{BitSet})

        t = tracer(JacobianTracer{BitSet}, 2)
        @test JacobianTracer(t) == t
        @test empty(t) == empty(JacobianTracer{BitSet})
        @test JacobianTracer{BitSet}(1) == empty(JacobianTracer{BitSet})

        # Base.show
        @test_reference "references/show/ConnectivityTracer.txt" repr(
            "text/plain", tracer(ConnectivityTracer{BitSet}, 2)
        )
        @test_reference "references/show/JacobianTracer.txt" repr(
            "text/plain", tracer(JacobianTracer{BitSet}, 2)
        )
    end
    @testset "Second order" begin
        @test pattern(identity, HessianTracer{BitSet}, rand()) ≈ [0;;]
        @test pattern(sqrt, HessianTracer{BitSet}, rand()) ≈ [1;;]

        @test pattern(x -> 1 * x, HessianTracer{BitSet}, rand()) ≈ [0;;]
        @test pattern(x -> x * 1, HessianTracer{BitSet}, rand()) ≈ [0;;]

        # Code coverage
        @test pattern(typemax, HessianTracer{BitSet}, 1) ≈ [0;;]
        @test pattern(x -> x^(2im), HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> (2im)^x, HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> x^(2//3), HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> (2//3)^x, HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> x^ℯ, HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> ℯ^x, HessianTracer{BitSet}, 1) ≈ [1;;]
        @test pattern(x -> round(x, RoundNearestTiesUp), HessianTracer{BitSet}, 1) ≈ [0;;]

        @test rand(HessianTracer{BitSet}) == empty(HessianTracer{BitSet})

        t = tracer(HessianTracer{BitSet}, 2)
        @test HessianTracer(t) == t
        @test empty(t) == empty(HessianTracer{BitSet})
        @test HessianTracer{BitSet}(1) == empty(HessianTracer{BitSet})

        x = rand(4)

        f(x) = x[1] / x[2] + x[3] / 1 + 1 / x[4]
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 1 0 0
            1 1 0 0
            0 0 0 0
            0 0 0 1
        ]

        f(x) = x[1] * x[2] + x[3] * 1 + 1 * x[4]
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 1 0 0
            1 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        f(x) = (x[1] * x[2]) * (x[3] * x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 1 1 1
            1 0 1 1
            1 1 0 1
            1 1 1 0
        ]

        f(x) = (x[1] + x[2]) * (x[3] + x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 0 1 1
            0 0 1 1
            1 1 0 0
            1 1 0 0
        ]

        f(x) = (x[1] + x[2] + x[3] + x[4])^2
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            1 1 1 1
            1 1 1 1
            1 1 1 1
            1 1 1 1
        ]

        f(x) = 1 / (x[1] + x[2] + x[3] + x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            1 1 1 1
            1 1 1 1
            1 1 1 1
            1 1 1 1
        ]

        f(x) = (x[1] - x[2]) + (x[3] - 1) + (1 - x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 0 0 0
            0 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        f(x) = copysign(x[1] * x[2], x[3] * x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 1 0 0
            1 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        f(x) = div(x[1] * x[2], x[3] * x[4])
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 0 0 0
            0 0 0 0
            0 0 0 0
            0 0 0 0
        ]

        x = rand()
        f(x) = sum(sincosd(x))
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [1;;]

        x = rand(4)
        f(x) = sum(diff(x) .^ 3)
        H = pattern(f, HessianTracer{BitSet}, x)
        @test H ≈ [
            1 1 0 0
            1 1 1 0
            0 1 1 1
            0 0 1 1
        ]

        x = rand(5)
        foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
        H = pattern(foo, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 0 0 0 0
            0 0 1 0 0
            0 1 0 0 0
            0 0 0 1 0
            0 0 0 0 0
        ]

        bar(x) = foo(x) + x[2]^x[5]
        H = pattern(bar, HessianTracer{BitSet}, x)
        @test H ≈ [
            0 0 0 0 0
            0 1 1 0 1
            0 1 0 0 0
            0 0 0 1 0
            0 1 0 0 1
        ]

        # Base.show
        @test_reference "references/show/HessianTracer.txt" repr(
            "text/plain", tracer(HessianTracer{BitSet}, 2)
        )
    end
    @testset "Real-world tests" begin
        @testset "NNlib" begin
            x = rand(3, 3, 2, 1) # WHCN
            w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
            C = pattern(x -> NNlib.conv(x, w), ConnectivityTracer{BitSet}, x)
            @test_reference "references/pattern/connectivity/NNlib/conv.txt" BitMatrix(C)
            J = pattern(x -> NNlib.conv(x, w), JacobianTracer{BitSet}, x)
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

            C = pattern(f!, du, ConnectivityTracer{BitSet}, u)
            @test_reference "references/pattern/connectivity/Brusselator.txt" BitMatrix(C)
            J = pattern(f!, du, JacobianTracer{BitSet}, u)
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
