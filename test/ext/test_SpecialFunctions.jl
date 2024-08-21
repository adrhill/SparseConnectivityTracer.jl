
using SparseConnectivityTracer
using SpecialFunctions: erf, beta
using Test

# Load definitions of GRADIENT_TRACERS, GRADIENT_PATTERNS, HESSIAN_TRACERS and HESSIAN_PATTERNS
include("../tracers_definitions.jl")

@testset "Jacobian Global" begin
    method = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, method)

    @test J(x -> erf(x[1]), rand(2)) == [1 0]
    @test J(x -> beta(x[1], x[2]), rand(3)) == [1 1 0]
end

# TODO: add tests
# @testset "Jacobian Local" begin
#     method = TracerLocalSparsityDetector()
#     J(f, x) = jacobian_sparsity(f, x, method)
# end

@testset "Global Hessian" begin
    method = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, method)

    @test H(x -> erf(x[1]), rand(2)) == [
        1 0
        0 0
    ]
    @test H(x -> beta(x[1], x[2]), rand(3)) == [
        1 1 0
        1 1 0
        0 0 0
    ]
end

# TODO: add tests
# @testset "Local Hessian" begin
#     method = TracerLocalSparsityDetector()
#     H(f, x) = hessian_sparsity(f, x, method)
# end
