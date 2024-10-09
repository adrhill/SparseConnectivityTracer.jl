using Pkg
Pkg.develop(;
    path=joinpath(@__DIR__, "..", "benchmark", "SparseConnectivityTracerBenchmarks")
)

using SparseConnectivityTracer
using Documenter: Documenter, DocMeta
using Test

DocMeta.setdocmeta!(
    SparseConnectivityTracer,
    :DocTestSetup,
    :(using SparseConnectivityTracer);
    recursive=true,
)

GROUP = get(ENV, "JULIA_SCT_TEST_GROUP", "Core")

@testset verbose = true "SparseConnectivityTracer.jl" begin
    if GROUP in ("Core", "All")
        @testset verbose = true "Linting" begin
            @info "Testing linting..."
            include("linting.jl")
        end
    end
    if GROUP in ("Core", "All")
        @testset "Doctests" begin
            Documenter.doctest(SparseConnectivityTracer)
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
        @testset "Operator classification" begin
            include("classification.jl")
        end
    end

    if GROUP in ("Core", "All")
        @info "Testing simple examples..."
        @testset verbose = true "Simple examples" begin
            @testset "Tracer Construction" begin
                include("test_constructors.jl")
            end
            @testset "GradientTracer" begin
                include("test_gradient.jl")
            end
            @testset "HessianTracer" begin
                include("test_hessian.jl")
            end
            @testset "Array overloads" begin
                include("test_arrays.jl")
            end
            @testset "ComponentArrays" begin
                include("componentarrays.jl")
            end
        end
    end
    if GROUP in ("Core", "All")
        @info "Testing package extensions..."
        @testset verbose = true "Package extensions" begin
            for ext in (:LogExpFunctions, :NaNMath, :NNlib, :SpecialFunctions)
                @testset "$ext" begin
                    @info "...$ext"
                    include("ext/test_$ext.jl")
                end
            end
            # Some extensions are only loaded in newer Julia releases
            for ext in (:DataInterpolations,)
                @testset "$ext" begin
                    @info "...$ext"
                    include("ext/test_$ext.jl")
                end
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

    if GROUP in ("Benchmarks", "All")
        @info "Testing benchmarks correctness..."
        @testset "Benchmarks correctness" begin
            include("benchmarks_correctness.jl")
        end
    end

    if GROUP in ("NLPModels", "All")
        @info "Testing NLPModels..."
        @testset "NLPModels" begin
            include("nlpmodels.jl")
        end
    end
end
