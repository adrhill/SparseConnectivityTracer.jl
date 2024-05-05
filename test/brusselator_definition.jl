using ReferenceTests
using SparseConnectivityTracer
using Test

# Brusselator example taken from Gowda et al.
# "Sparsity Programming: Automated Sparsity-Aware Optimizations in Differentiable Programming"
# https://openreview.net/pdf?id=rJlPdcY38B

#! format: off
brusselator_f(x, y, t) = ifelse((((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) && (t >= 1.1), 5., 0.)
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a
function brusselator_2d_loop(du, u, p, t)
    A, B, alpha, xyd, dx, N = p; alpha = alpha/dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd[I[1]], xyd[I[2]]
        ip1, im1, jp1, jm1 = limit(i+1, N), limit(i-1, N), limit(j+1, N), limit(j-1, N)
        du[i,j,1] = alpha*(u[im1,j,1] + u[ip1,j,1] + u[i,jp1,1] + u[i,jm1,1] - 4u[i,j,1]) +
        B + u[i,j,1]^2*u[i,j,2] - (A + 1)*u[i,j,1] + brusselator_f(x, y, t)
        du[i,j,2] = alpha*(u[im1,j,2] + u[ip1,j,2] + u[i,jp1,2] + u[i,jm1,2] - 4u[i,j,2]) +
        A*u[i,j,1] - u[i,j,1]^2*u[i,j,2]
    end
end
#! format: on

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
