using SparseConnectivityTracer, ForwardDiff
using ForwardDiff: Dual, Partials, Tag

using Test

d = Dual{Tag{*,Float64}}(1.2, 3.4)
@testset "$D" for D in (TracerSparsityDetector, TracerLocalSparsityDetector)
    detector = D()
    # Testing on multiplication ensures that methods from Base have been overloaded,
    # Since this would otherwise throw an ambiguity error:
    # https://github.com/adrhill/SparseConnectivityTracer.jl/issues/196
    @testset "Jacobian" begin
        @test jacobian_sparsity(x -> x * d, 1.0, detector) ≈ [1;;]
        @test jacobian_sparsity(x -> d * x, 1.0, detector) ≈ [1;;]
    end
    @testset "Hessian" begin
        @test hessian_sparsity(x -> x * d, 1.0, detector) ≈ [0;;]
        @test hessian_sparsity(x -> d * x, 1.0, detector) ≈ [0;;]
    end
end
