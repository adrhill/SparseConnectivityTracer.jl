using ReferenceTests
using SparseConnectivityTracer
using Test

include("brusselator_definition.jl")

@testset "Set type $S" for S in (BitSet, Set{UInt64}, SortedVector{UInt64})
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

    C = connectivity_pattern(f!, du, u, S)
    @test_reference "references/pattern/connectivity/Brusselator.txt" BitMatrix(C)
    J = jacobian_pattern(f!, du, u, S)
    @test_reference "references/pattern/jacobian/Brusselator.txt" BitMatrix(J)
    @test C == J

    C_ref = Symbolics.jacobian_sparsity(f!, du, u)
    @test C == C_ref
end
