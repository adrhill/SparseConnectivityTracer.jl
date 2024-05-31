using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector, product
using Test

@testset "$(typeof(S))" for S in (
    BitSet, Set{Int64}, DuplicateVector{Int64}, RecursiveSet{Int64}, SortedVector{Int64}
)
    x = S.(1:10)
    y = (x[1] ∪ x[3]) ∪ (x[3] ∪ ((x[5] ∪ x[7]) ∪ x[1]))

    @test sort(collect(y)) == [1, 3, 5, 7]
    @test length(collect(product(y, y))) == 16
end
