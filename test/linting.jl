using SparseConnectivityTracer
using Test

using Aqua: Aqua
using JET: JET
using ExplicitImports: ExplicitImports

# Load package extensions so they get tested by ExplicitImports.jl
using DataInterpolations: DataInterpolations
using NaNMath: NaNMath
using NNlib: NNlib
using SpecialFunctions: SpecialFunctions

@testset "Aqua tests" begin
    @info "...with Aqua.jl"
    Aqua.test_all(
        SparseConnectivityTracer;
        ambiguities = false,
        deps_compat = (check_extras = false,),
        persistent_tasks = false,
    )
end

if VERSION < v"1.12"
    # JET v0.9  is compatible with Julia <1.12
    # JET v0.10 is compatible with Julia ≥1.12
    # TODO: Update when 1.12 releases
    @testset "JET tests" begin
        @info "...with JET.jl"
        JET.test_package(SparseConnectivityTracer; target_defined_modules = true)
    end
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
            ignore = (
                # Used in code generation, which ExplicitImports doesn't pick up
                :AbstractTracer,
                :AkimaInterpolation,
                :BSplineApprox,
                :BSplineInterpolation,
                :ConstantInterpolation,
                :CubicHermiteSpline,
                :CubicSpline,
                :LagrangeInterpolation,
                :LinearInterpolation,
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
