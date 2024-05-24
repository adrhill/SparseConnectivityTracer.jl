using BenchmarkTools

using ADTypes: hessian_sparsity
using SparseConnectivityTracer

using SparseArrays: sprand

function hessbench(::Type{S}) where {S}
    suite = BenchmarkGroup()
    return suite
end
