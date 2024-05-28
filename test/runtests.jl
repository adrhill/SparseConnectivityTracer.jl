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

GROUP = get(ENV, "JULIA_SCT_TEST_GROUP", "All")

@testset verbose = true "SparseConnectivityTracer.jl" begin
    if GROUP in ("Core", "All")
        @testset verbose = true "Formalities" begin
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

    @testset verbose = true "Set types" begin
        if GROUP in ("Core", "All")
            @testset "BitSet" begin
                include("settypes/bitset.jl")
            end
            @testset "Set" begin
                include("settypes/set.jl")
            end
            @testset "SortedVector" begin
                include("settypes/sortedvector.jl")
            end
            @testset "DuplicateVector" begin
                include("settypes/duplicatevector.jl")
            end
            @testset "RecursiveSet" begin
                include("settypes/recursiveset.jl")
            end
            @testset "SortedVector" begin
                include("settypes/sortedvector.jl")
            end
        end
    end

    @testset "Classification of operators by diff'ability" begin
        if GROUP in ("Core", "All")
            include("classification.jl")
        end
    end

    @testset verbose = true "Simple examples" begin
        if GROUP in ("Core", "All")
            @testset "Tracer Construction" begin
                include("test_connectivity.jl")
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

    @testset verbose = true "Real-world examples" begin
        if GROUP in ("Core", "All")
            @testset "Brusselator" begin
                include("brusselator.jl")
            end
            @testset "Flux.jl" begin
                include("flux.jl")
            end
        end
        if GROUP in ("NLPModels", "All")
            @testset "NLPModels" begin
                include("nlpmodels.jl")
            end
        end
    end

    @testset "ADTypes integration" begin
        GROUP in ("Core", "All") && include("adtypes.jl")
    end
end
