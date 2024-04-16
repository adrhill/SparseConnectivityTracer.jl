using ADTypes
using SparseConnectivityTracer
using SparseArrays
using Test

sd = TracerSparsityDetector()

x = rand(10)
y = zeros(9)
J1 = ADTypes.jacobian_sparsity(diff, x, sd)
J2 = ADTypes.jacobian_sparsity((y, x) -> y .= diff(x), y, x, sd)
@test J1 == J2
@test J1 isa SparseMatrixCSC
@test J2 isa SparseMatrixCSC
@test nnz(J1) == nnz(J2) == 18
@test_throws ErrorException ADTypes.hessian_sparsity(sum, x, sd)
