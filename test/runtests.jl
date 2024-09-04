using Pkg
Pkg.develop(;
    path=joinpath(@__DIR__, "..", "benchmark", "SparseConnectivityTracerBenchmarks")
)

using SparseConnectivityTracer
using Compat: pkgversion
using Test

using JuliaFormatter: JuliaFormatter
using Aqua: Aqua
using JET: JET
using ExplicitImports: ExplicitImports
using Documenter: Documenter, DocMeta

# Load package extensions so they get tested by ExplicitImports.jl
using DataInterpolations: DataInterpolations
using NaNMath: NaNMath
using NNlib: NNlib
using SpecialFunctions: SpecialFunctions

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
                    @info "...with JuliaFormatter.jl"
                    @test JuliaFormatter.format(
                        SparseConnectivityTracer; verbose=false, overwrite=false
                    )
                end
                @testset "Aqua tests" begin
                    @info "...with Aqua.jl"
                    Aqua.test_all(
                        SparseConnectivityTracer;
                        ambiguities=false,
                        deps_compat=(check_extras=false,),
                        stale_deps=(ignore=[:Requires],),
                        persistent_tasks=false,
                    )
                end
                @testset "JET tests" begin
                    @info "...with JET.jl"
                    JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
                end
                @testset "ExplicitImports tests" begin
                    @info "...with ExplicitImports.jl"
                    @testset "Improper implicit imports" begin
                        @test ExplicitImports.check_no_implicit_imports(
                            SparseConnectivityTracer
                        ) === nothing
                    end
                    @testset "Improper explicit imports" begin
                        @test ExplicitImports.check_no_stale_explicit_imports(
                            SparseConnectivityTracer;
                            ignore=(
                                :AbstractTracer,
                                :AkimaInterpolation,
                                :BSplineApprox,
                                :BSplineInterpolation,
                                :CubicHermiteSpline,
                                :CubicSpline,
                                :LagrangeInterpolation,
                                :QuadraticInterpolation,
                                :QuadraticSpline,
                                :QuinticHermiteSpline,
                            ),
                        ) === nothing
                        @test ExplicitImports.check_all_explicit_imports_via_owners(
                            SparseConnectivityTracer
                        ) === nothing
                        # TODO: test in the future when `public` is more common
                        # @test ExplicitImports.check_all_explicit_imports_are_public(
                        #     SparseConnectivityTracer
                        # ) === nothing
                    end
                    @testset "Improper qualified accesses" begin
                        @test ExplicitImports.check_all_qualified_accesses_via_owners(
                            SparseConnectivityTracer
                        ) === nothing
                        @test ExplicitImports.check_no_self_qualified_accesses(
                            SparseConnectivityTracer
                        ) === nothing
                        # TODO: test in the future when `public` is more common
                        # @test ExplicitImports.check_all_qualified_accesses_are_public(
                        #     SparseConnectivityTracer
                        # ) === nothing
                    end
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
            for ext in (:NNlib, :SpecialFunctions, :LogExpFunctions, :NaNMath)
                @testset "$ext" begin
                    @info "...$ext"
                    include("ext/test_$ext.jl")
                end
            end
            # Some extensions are only loaded in newer Julia releases
            if VERSION >= v"1.10"
                for ext in (:DataInterpolations,)
                    @testset "$ext" begin
                        @info "...$ext"
                        include("ext/test_$ext.jl")
                    end
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
