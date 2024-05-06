using ADTypes
using ADTypes: AbstractSparsityDetector
using ReferenceTests
using SparseConnectivityTracer
using SparseConnectivityTracer: SortedVector
using Test

include("brusselator_definition.jl")

function test_brusselator(method::AbstractSparsityDetector)
    N = 6
    dims = (N, N, 2)
    A = 1.0
    B = 1.0
    alpha = 1.0
    xyd = fill(1.0, N)
    dx = 1.0
    p = (A, B, alpha, xyd, dx, N)

    u = rand(dims...)
    du = similar(u)
    f!(du, u) = brusselator_2d_loop(du, u, p, nothing)

    J = ADTypes.jacobian_sparsity(f!, du, u, method)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
end

@testset "$method" for method in (
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
)
    test_brusselator(method)
end
