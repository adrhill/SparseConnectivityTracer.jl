using SparseConnectivityTracer: ×
using Test

x = Set.(1:10)
y = (x[1] ∪ x[3]) ∪ (x[3] ∪ ((x[5] ∪ x[7]) ∪ x[1]))

@test sort(collect(y)) == [1, 3, 5, 7]
@test y × y isa Set{Tuple{Int,Int}}
@test length(collect(y × y)) == 16
