using SparseConnectivityTracer

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

@testset verbose = true "SparseConnectivityTracer.jl" begin
    @testset verbose = true "Formalities" begin
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
                persistent_tasks=false,
            )
        end
        @testset "JET tests" begin
            JET.test_package(SparseConnectivityTracer; target_defined_modules=true)
        end
        @testset "Doctests" begin
            Documenter.doctest(SparseConnectivityTracer)
        end
    end

    @testset verbose = true "Set types" begin
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

    @testset "Classification of operators by diff'ability" begin
        include("classification.jl")
    end

    @testset verbose = true "Simple examples" begin
        @testset "First order" begin
            include("first_order.jl")
        end
        @testset "Second order" begin
            include("second_order.jl")
        end
    end

    @testset verbose = true "Real-world examples" begin
        @testset "Brusselator" begin
            include("brusselator.jl")
        end
        @testset "Flux.jl" begin
            include("flux.jl")
        end
    end

    @testset "ADTypes integration" begin
        include("adtypes.jl")
    end
end
