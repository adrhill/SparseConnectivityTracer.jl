using SparseConnectivityTracer: DuplicateVector, ×
using Test

x = DuplicateVector{Int}.(1:10)
y = (x[1] ∪ x[3]) ∪ (x[3] ∪ ((x[5] ∪ x[7]) ∪ x[1]))

@test startswith(string(y), "DuplicateVector(")
@test sort(collect(y)) == [1, 3, 5, 7]
@test y × y isa DuplicateVector
@test length(collect(y × y)) == 16
