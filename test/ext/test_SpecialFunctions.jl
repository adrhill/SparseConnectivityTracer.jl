using SparseConnectivityTracer
using SpecialFunctions: erf, beta
using Test

# Load definitions of GRADIENT_TRACERS and HESSIAN_TRACERS
include("../tracers_definitions.jl")

@testset "Jacobian Global" begin
    detector = TracerSparsityDetector()
    J(f, x) = jacobian_sparsity(f, x, detector)

    @test J(x -> erf(x[1]), rand(2)) == [1 0]
    @test J(x -> beta(x[1], x[2]), rand(3)) == [1 1 0]
end

# TODO: add tests
# @testset "Jacobian Local" begin
#     detector = TracerLocalSparsityDetector()
#     J(f, x) = jacobian_sparsity(f, x, detector)
# end

@testset "Global Hessian" begin
    detector = TracerSparsityDetector()
    H(f, x) = hessian_sparsity(f, x, detector)

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
#     detector = TracerLocalSparsityDetector()
#     H(f, x) = hessian_sparsity(f, x, detector)
# end
