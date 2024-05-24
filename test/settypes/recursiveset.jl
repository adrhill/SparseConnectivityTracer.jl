using SparseConnectivityTracer: RecursiveSet, ×
using Test

x = RecursiveSet{Int}.(1:10)
y = (x[1] ∪ x[3]) ∪ (x[3] ∪ ((x[5] ∪ x[7]) ∪ x[1]))

@test startswith(string(y), "RecursiveSet")
@test sort(collect(y)) == [1, 3, 5, 7]
@test y × y isa RecursiveSet
@test length(collect(y × y)) == 16
