using SparseConnectivityTracer
using Test

using JuliaFormatter: JuliaFormatter
using Aqua: Aqua
using JET: JET
using ExplicitImports: ExplicitImports

# Load package extensions so they get tested by ExplicitImports.jl
using ForwardDiff: ForwardDiff
using DataInterpolations: DataInterpolations
using NaNMath: NaNMath
using NNlib: NNlib
using SpecialFunctions: SpecialFunctions

@testset "Code formatting" begin
    @info "...with JuliaFormatter.jl"
    @test JuliaFormatter.format(SparseConnectivityTracer; verbose=false, overwrite=false)
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
        @test ExplicitImports.check_no_implicit_imports(SparseConnectivityTracer) ===
            nothing
    end
    @testset "Improper explicit imports" begin
        @test ExplicitImports.check_no_stale_explicit_imports(
            SparseConnectivityTracer;
            ignore=(
                # Used in code generation, which ExplicitImports doesn't pick up
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
        @test ExplicitImports.check_no_self_qualified_accesses(SparseConnectivityTracer) ===
            nothing
        # TODO: test in the future when `public` is more common
        # @test ExplicitImports.check_all_qualified_accesses_are_public(
        #     SparseConnectivityTracer
        # ) === nothing
    end
end
