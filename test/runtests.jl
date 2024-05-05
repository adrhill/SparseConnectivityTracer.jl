using SparseConnectivityTracer
using SparseConnectivityTracer: tracer, trace_input, inputs, empty

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
    @testset verbose = true "Set types" begin
        @testset "SortedVector" begin
            include("sortedvector.jl")
        end
        @testset "RecursiveSet" begin
            include("recursiveset.jl")
        end
    end
    @testset "Classification of operators by diff'ability" begin
        include("test_differentiability.jl")
    end
    @testset "First order" begin
        for S in (BitSet, Set{UInt64}, SortedVector{UInt64})
            @testset "Set type $S" begin
                CT = ConnectivityTracer{S}
                JT = JacobianTracer{S}

                x = rand(3)
                xt = trace_input(CT, x)

                # Matrix multiplication
                A = rand(1, 3)
                yt = only(A * xt)
                @test connectivity_pattern(x -> only(A * x), x, S) ≈ [1 1 1]

                # Custom functions
                f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
                yt = f(xt)

                @test connectivity_pattern(f, x, S) ≈ [1 0 0; 1 1 0; 0 0 1]
                @test jacobian_pattern(f, x, S) ≈ [1 0 0; 1 1 0; 0 0 1]

                @test connectivity_pattern(identity, rand(), S) ≈ [1;;]
                @test jacobian_pattern(identity, rand(), S) ≈ [1;;]
                @test connectivity_pattern(Returns(1), 1, S) ≈ [0;;]
                @test jacobian_pattern(Returns(1), 1, S) ≈ [0;;]

                # Test JacobianTracer on functions with zero derivatives
                x = rand(2)
                g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
                @test connectivity_pattern(g, x, S) ≈ [1 1; 1 1; 1 1]
                @test jacobian_pattern(g, x, S) ≈ [1 1; 0 0; 1 0]

                # Code coverage
                @test connectivity_pattern(x -> [sincos(x)...], 1, S) ≈ [1; 1]
                @test connectivity_pattern(typemax, 1, S) ≈ [0;;]
                @test connectivity_pattern(x -> x^(2//3), 1, S) ≈ [1;;]
                @test connectivity_pattern(x -> (2//3)^x, 1, S) ≈ [1;;]
                @test connectivity_pattern(x -> x^ℯ, 1, S) ≈ [1;;]
                @test connectivity_pattern(x -> ℯ^x, 1, S) ≈ [1;;]
                @test connectivity_pattern(x -> round(x, RoundNearestTiesUp), 1, S) ≈ [1;;]

                @test jacobian_pattern(x -> [sincos(x)...], 1, S) ≈ [1; 1]
                @test jacobian_pattern(typemax, 1, S) ≈ [0;;]
                @test jacobian_pattern(x -> x^(2//3), 1, S) ≈ [1;;]
                @test jacobian_pattern(x -> (2//3)^x, 1, S) ≈ [1;;]
                @test jacobian_pattern(x -> x^ℯ, 1, S) ≈ [1;;]
                @test jacobian_pattern(x -> ℯ^x, 1, S) ≈ [1;;]
                @test jacobian_pattern(x -> round(x, RoundNearestTiesUp), 1, S) ≈ [0;;]

                # Base.show
                @test_reference "references/show/ConnectivityTracer_$S.txt" repr(
                    "text/plain", tracer(CT, 2)
                )
                @test_reference "references/show/JacobianTracer_$S.txt" repr(
                    "text/plain", tracer(JT, 2)
                )
            end
        end
    end
    @testset "Second order" begin
        for S in (BitSet, Set{UInt64}, SortedVector{UInt64})
            @testset "Set type $S" begin
                HT = HessianTracer{S}

                @test hessian_pattern(identity, rand(), S) ≈ [0;;]
                @test hessian_pattern(sqrt, rand(), S) ≈ [1;;]

                @test hessian_pattern(x -> 1 * x, rand(), S) ≈ [0;;]
                @test hessian_pattern(x -> x * 1, rand(), S) ≈ [0;;]

                # Code coverage
                @test hessian_pattern(typemax, 1) ≈ [0;;]
                @test hessian_pattern(x -> x^(2im), 1) ≈ [1;;]
                @test hessian_pattern(x -> (2im)^x, 1) ≈ [1;;]
                @test hessian_pattern(x -> x^(2//3), 1) ≈ [1;;]
                @test hessian_pattern(x -> (2//3)^x, 1) ≈ [1;;]
                @test hessian_pattern(x -> x^ℯ, 1) ≈ [1;;]
                @test hessian_pattern(x -> ℯ^x, 1) ≈ [1;;]
                @test hessian_pattern(x -> round(x, RoundNearestTiesUp), 1) ≈ [0;;]

                H = hessian_pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], rand(4), S)
                @test H ≈ [
                    0 1 0 0
                    1 1 0 0
                    0 0 0 0
                    0 0 0 1
                ]

                H = hessian_pattern(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], rand(4), S)
                @test H ≈ [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = hessian_pattern(x -> (x[1] * x[2]) * (x[3] * x[4]), rand(4), S)
                @test H ≈ [
                    0 1 1 1
                    1 0 1 1
                    1 1 0 1
                    1 1 1 0
                ]

                H = hessian_pattern(x -> (x[1] + x[2]) * (x[3] + x[4]), rand(4), S)
                @test H ≈ [
                    0 0 1 1
                    0 0 1 1
                    1 1 0 0
                    1 1 0 0
                ]

                H = hessian_pattern(x -> (x[1] + x[2] + x[3] + x[4])^2, rand(4), S)
                @test H ≈ [
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                ]

                H = hessian_pattern(x -> 1 / (x[1] + x[2] + x[3] + x[4]), rand(4), S)
                @test H ≈ [
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                ]

                H = hessian_pattern(
                    x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), rand(4), S
                )
                @test H ≈ [
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = hessian_pattern(x -> copysign(x[1] * x[2], x[3] * x[4]), rand(4), S)
                @test H ≈ [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = hessian_pattern(x -> div(x[1] * x[2], x[3] * x[4]), rand(4), S)
                @test H ≈ [
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = hessian_pattern(x -> sum(sincosd(x)), 1.0, S)
                @test H ≈ [1;;]

                H = hessian_pattern(x -> sum(diff(x) .^ 3), rand(4), S)
                @test H ≈ [
                    1 1 0 0
                    1 1 1 0
                    0 1 1 1
                    0 0 1 1
                ]

                x = rand(5)
                foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
                H = hessian_pattern(foo, x, S)
                @test H ≈ [
                    0 0 0 0 0
                    0 0 1 0 0
                    0 1 0 0 0
                    0 0 0 1 0
                    0 0 0 0 0
                ]

                bar(x) = foo(x) + x[2]^x[5]
                H = hessian_pattern(bar, x, S)
                @test H ≈ [
                    0 0 0 0 0
                    0 1 1 0 1
                    0 1 0 0 0
                    0 0 0 1 0
                    0 1 0 0 1
                ]

                # Base.show
                @test_reference "references/show/HessianTracer_$S.txt" repr(
                    "text/plain", tracer(HT, 2)
                )
            end
        end
    end
    @testset "Real-world tests" begin
        include("brusselator.jl")
        for S in (BitSet, Set{UInt64}, SortedVector{UInt64})
            @testset "Set type $S" begin
                @testset "Brusselator" begin
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

                    C = connectivity_pattern(f!, du, u, S)
                    @test_reference "references/pattern/connectivity/Brusselator.txt" BitMatrix(
                        C
                    )
                    J = jacobian_pattern(f!, du, u, S)
                    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(
                        J
                    )
                    @test C == J

                    C_ref = Symbolics.jacobian_sparsity(f!, du, u)
                    @test C == C_ref
                end
                @testset "NNlib" begin
                    x = rand(3, 3, 2, 1) # WHCN
                    w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
                    C = jacobian_pattern(x -> NNlib.conv(x, w), x, S)
                    @test_reference "references/pattern/connectivity/NNlib/conv.txt" BitMatrix(
                        C
                    )
                    J = jacobian_pattern(x -> NNlib.conv(x, w), x, S)
                    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(
                        J
                    )
                    @test C == J
                end
            end
        end
    end
    @testset "ADTypes integration" begin
        include("adtypes.jl")
    end
end
