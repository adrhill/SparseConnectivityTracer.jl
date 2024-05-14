using ADTypes: jacobian_sparsity, hessian_sparsity
using SparseConnectivityTracer
using SparseArrays
using Test

sd = TracerSparsityDetector(BitSet, Set{Tuple{Int,Int}})

x = rand(10)
y = zeros(9)
J1 = jacobian_sparsity(diff, x, sd)
J2 = jacobian_sparsity((y, x) -> y .= diff(x), y, x, sd)
@test J1 == J2
@test J1 isa SparseMatrixCSC{Bool,Int}
@test J2 isa SparseMatrixCSC{Bool,Int}
@test nnz(J1) == nnz(J2) == 18

H1 = hessian_sparsity(x -> sum(diff(x)), x, sd)
@test H1 ≈ zeros(10, 10)

x = rand(5)
f(x) = x[1] + x[2] * x[3] + 1 / x[4] + 1 * x[5]
H2 = hessian_sparsity(f, x, sd)
@test H2 isa SparseMatrixCSC{Bool,Int}
@test H2 ≈ [
    0 0 0 0 0
    0 0 1 0 0
    0 1 0 0 0
    0 0 0 1 0
    0 0 0 0 0
]
