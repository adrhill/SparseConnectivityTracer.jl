using SparseConnectivityTracer, ForwardDiff
using ForwardDiff: Dual, Partials, Tag

using Test

d = Dual{Tag{*,Float64}}(1.2, 3.4)
@testset "$D" for D in (TracerSparsityDetector, TracerLocalSparsityDetector)
    detector = D()
    @test jacobian_sparsity(x -> x * d, 1.0, detector) ≈ [1;;]
    @test jacobian_sparsity(x -> d * x, 1.0, detector) ≈ [1;;]
end
