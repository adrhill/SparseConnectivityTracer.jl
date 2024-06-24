using Pkg
Pkg.develop(; path=joinpath(@__DIR__, "..", "libs", "SparseConnectivityTracerBenchmarks"))

using SparseConnectivityTracer

using Compat
using Test
using ReferenceTests
using JuliaFormatter
using Aqua
using JET
using Documenter

using LinearAlgebra
using Random

DocMeta.setdocmeta!(
    SparseConnectivityTracer,
    :DocTestSetup,
    :(using SparseConnectivityTracer);
    recursive=true,
)

GROUP = get(ENV, "JULIA_SCT_TEST_GROUP", "Core")

@testset verbose = true "SparseConnectivityTracer.jl" begin
    if GROUP in ("Core", "All")
        @testset verbose = true "Formalities" begin
            @info "Testing formalities..."
            if VERSION >= v"1.10"
                @testset "Code formatting" begin
                    @test JuliaFormatter.format(
                        SparseConnectivityTracer; verbose=false, overwrite=false
                    )
                end
                @testset "Aqua tests" begin
                    Aqua.test_all(
                        SparseConnectivityTracer;
                        ambiguities=false,
                        deps_compat=(ignore=[:Random, :SparseArrays], check_extras=false),
                        stale_deps=(ignore=[:Requires],),
                        persistent_tasks=false,
                    )
                end
                @testset "JET tests" begin
                    JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
                end
            end
            @testset "Doctests" begin
                Documenter.doctest(SparseConnectivityTracer)
            end
        end
    end

    if GROUP in ("Core", "All")
        @testset verbose = true "Set types" begin
            @testset "Correctness" begin
                include("settypes/correctness.jl")
            end
            @testset "SortedVector" begin
                include("settypes/sortedvector.jl")
            end
        end
    end

    if GROUP in ("Core", "All")
        @info "Testing operator classification..."
        @testset "Classification of operators by diff'ability" begin
            include("classification.jl")
        end
    end

    if GROUP in ("Core", "All")
        @info "Testing simple examples..."
        @testset verbose = true "Simple examples" begin
            @testset "Tracer Construction" begin
                include("test_constructors.jl")
            end
            @testset "ConnectivityTracer" begin
                include("test_connectivity.jl")
            end
            @testset "GradientTracer" begin
                include("test_gradient.jl")
            end
            @testset "HessianTracer" begin
                include("test_hessian.jl")
            end
        end
    end

    if GROUP in ("Core", "All")
        @info "Testing real-world examples..."
        @testset verbose = true "Real-world examples" begin
            @testset "Brusselator" begin
                include("brusselator.jl")
            end
            if pkgversion(NNlib) >= v"0.9.18" # contains NNlib PR #592
                @testset "Flux.jl" begin
                    include("flux.jl")
                end
            end
        end
    end

    if GROUP in ("Core", "All")
        @info "Testing ADTypes integration..."
        @testset "ADTypes integration" begin
            include("adtypes.jl")
        end
    end

    if GROUP in ("NLPModels", "All")
        @info "Testing NLPModels..."
        @testset "NLPModels" begin
            include("nlpmodels.jl")
        end
    end
end
