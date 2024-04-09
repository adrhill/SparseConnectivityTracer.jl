using Base: get_extension
using ForwardDiff: ForwardDiff
using SparseArrays
using SparseConnectivityTracer
using SparseDiffTools
using Test

ext = Base.get_extension(
    SparseConnectivityTracer, :SparseConnectivityTracerSparseDiffToolsExt
)
@test !isnothing(ext)

sd = ext.ConnectivityTracerSparsityDetection()
adtype = SparseDiffTools.AutoSparseForwardDiff()

x = rand(10)
y = zeros(9)
J1 = sparse_jacobian(adtype, sd, diff, x)
J2 = sparse_jacobian(adtype, sd, (y, x) -> y .= diff(x), y, x)
@test J1 == J2
@test J1 isa SparseMatrixCSC
@test J2 isa SparseMatrixCSC
@test nnz(J1) == nnz(J2) == 18
