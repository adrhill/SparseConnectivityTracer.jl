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
        for S in (BitSet, Set{UInt64})
            @testset "Set type $S" begin
                CT = ConnectivityTracer{S}
                JT = JacobianTracer{S}

                x = rand(3)
                xt = trace_input(CT, x)

                # Matrix multiplication
                A = rand(1, 3)
                yt = only(A * xt)
                @test pattern(x -> only(A * x), CT, x) ≈ [1 1 1]

                # Custom functions
                f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])]
                yt = f(xt)

                @test pattern(f, CT, x) ≈ [1 0 0; 1 1 0; 0 0 1]
                @test pattern(f, JT, x) ≈ [1 0 0; 1 1 0; 0 0 1]

                @test pattern(identity, CT, rand()) ≈ [1;;]
                @test pattern(identity, JT, rand()) ≈ [1;;]
                @test pattern(Returns(1), CT, 1) ≈ [0;;]
                @test pattern(Returns(1), JT, 1) ≈ [0;;]

                # Test JacobianTracer on functions with zero derivatives
                x = rand(2)
                g(x) = [x[1] * x[2], ceil(x[1] * x[2]), x[1] * round(x[2])]
                @test pattern(g, CT, x) ≈ [1 1; 1 1; 1 1]
                @test pattern(g, JT, x) ≈ [1 1; 0 0; 1 0]

                # Code coverage
                @test pattern(x -> [sincos(x)...], CT, 1) ≈ [1; 1]
                @test pattern(x -> [sincos(x)...], JT, 1) ≈ [1; 1]
                @test pattern(typemax, CT, 1) ≈ [0;;]
                @test pattern(typemax, JT, 1) ≈ [0;;]
                @test pattern(x -> x^(2//3), CT, 1) ≈ [1;;]
                @test pattern(x -> x^(2//3), JT, 1) ≈ [1;;]
                @test pattern(x -> (2//3)^x, CT, 1) ≈ [1;;]
                @test pattern(x -> (2//3)^x, JT, 1) ≈ [1;;]
                @test pattern(x -> x^ℯ, CT, 1) ≈ [1;;]
                @test pattern(x -> x^ℯ, JT, 1) ≈ [1;;]
                @test pattern(x -> ℯ^x, CT, 1) ≈ [1;;]
                @test pattern(x -> ℯ^x, JT, 1) ≈ [1;;]
                @test pattern(x -> round(x, RoundNearestTiesUp), CT, 1) ≈ [1;;]
                @test pattern(x -> round(x, RoundNearestTiesUp), JT, 1) ≈ [0;;]

                @test rand(CT) == empty(CT)
                @test rand(JT) == empty(JT)

                t = tracer(CT, 2)
                @test ConnectivityTracer(t) == t
                @test empty(t) == empty(CT)
                @test CT(1) == empty(CT)

                t = tracer(JT, 2)
                @test JacobianTracer(t) == t
                @test empty(t) == empty(JT)
                @test JT(1) == empty(JT)

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
        for S in (BitSet, Set{UInt64})
            @testset "Set type $S" begin
                HT = HessianTracer{S}
                @test pattern(identity, HT, rand()) ≈ [0;;]
                @test pattern(sqrt, HT, rand()) ≈ [1;;]

                @test pattern(x -> 1 * x, HT, rand()) ≈ [0;;]
                @test pattern(x -> x * 1, HT, rand()) ≈ [0;;]

                # Code coverage
                @test pattern(typemax, HT, 1) ≈ [0;;]
                @test pattern(x -> x^(2im), HT, 1) ≈ [1;;]
                @test pattern(x -> (2im)^x, HT, 1) ≈ [1;;]
                @test pattern(x -> x^(2//3), HT, 1) ≈ [1;;]
                @test pattern(x -> (2//3)^x, HT, 1) ≈ [1;;]
                @test pattern(x -> x^ℯ, HT, 1) ≈ [1;;]
                @test pattern(x -> ℯ^x, HT, 1) ≈ [1;;]
                @test pattern(x -> round(x, RoundNearestTiesUp), HT, 1) ≈ [0;;]

                @test rand(HT) == empty(HT)

                t = tracer(HT, 2)
                @test HessianTracer(t) == t
                @test empty(t) == empty(HT)
                @test HT(1) == empty(HT)

                H = pattern(x -> x[1] / x[2] + x[3] / 1 + 1 / x[4], HT, rand(4))
                @test H ≈ [
                    0 1 0 0
                    1 1 0 0
                    0 0 0 0
                    0 0 0 1
                ]

                H = pattern(x -> x[1] * x[2] + x[3] * 1 + 1 * x[4], HT, rand(4))
                @test H ≈ [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = pattern(x -> (x[1] * x[2]) * (x[3] * x[4]), HT, rand(4))
                @test H ≈ [
                    0 1 1 1
                    1 0 1 1
                    1 1 0 1
                    1 1 1 0
                ]

                H = pattern(x -> (x[1] + x[2]) * (x[3] + x[4]), HT, rand(4))
                @test H ≈ [
                    0 0 1 1
                    0 0 1 1
                    1 1 0 0
                    1 1 0 0
                ]

                H = pattern(x -> (x[1] + x[2] + x[3] + x[4])^2, HT, rand(4))
                @test H ≈ [
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                ]

                H = pattern(x -> 1 / (x[1] + x[2] + x[3] + x[4]), HT, rand(4))
                @test H ≈ [
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                    1 1 1 1
                ]

                H = pattern(x -> (x[1] - x[2]) + (x[3] - 1) + (1 - x[4]), HT, rand(4))
                @test H ≈ [
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = pattern(x -> copysign(x[1] * x[2], x[3] * x[4]), HT, rand(4))
                @show H
                @test H ≈ [
                    0 1 0 0
                    1 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = pattern(x -> div(x[1] * x[2], x[3] * x[4]), HT, rand(4))
                @test H ≈ [
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                    0 0 0 0
                ]

                H = pattern(x -> sum(sincosd(x)), HT, 1.0)
                @test H ≈ [1;;]

                H = pattern(x -> sum(diff(x) .^ 3), HT, rand(4))
                @test H ≈ [
                    1 1 0 0
                    1 1 1 0
                    0 1 1 1
                    0 0 1 1
                ]

                x = rand(5)
                foo(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
                H = pattern(foo, HT, x)
                @test H ≈ [
                    0 0 0 0 0
                    0 0 1 0 0
                    0 1 0 0 0
                    0 0 0 1 0
                    0 0 0 0 0
                ]

                bar(x) = foo(x) + x[2]^x[5]
                H = pattern(bar, HT, x)
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
        for S in (BitSet, Set{UInt64})
            @testset "Set type $S" begin
                CT = ConnectivityTracer{S}
                JT = JacobianTracer{S}
                HT = HessianTracer{S}

                @testset "NNlib" begin
                    x = rand(3, 3, 2, 1) # WHCN
                    w = rand(2, 2, 2, 1) # Conv((2, 2), 2 => 1)
                    C = pattern(x -> NNlib.conv(x, w), CT, x)
                    @test_reference "references/pattern/connectivity/NNlib/conv.txt" BitMatrix(
                        C
                    )
                    J = pattern(x -> NNlib.conv(x, w), JT, x)
                    @test_reference "references/pattern/jacobian/NNlib/conv.txt" BitMatrix(
                        J
                    )
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

                    C = pattern(f!, du, CT, u)
                    @test_reference "references/pattern/connectivity/Brusselator.txt" BitMatrix(
                        C
                    )
                    J = pattern(f!, du, JT, u)
                    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(
                        J
                    )
                    @test C == J

                    C_ref = Symbolics.jacobian_sparsity(f!, du, u)
                    @test C == C_ref
                end
            end
        end
    end
    @testset "ADTypes integration" begin
        include("adtypes.jl")
    end
end
