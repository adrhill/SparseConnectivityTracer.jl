using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector, product
using Test

@testset "$S" for S in (
    BitSet, Set{Int}, DuplicateVector{Int}, RecursiveSet{Int}, SortedVector{Int}
)
    x = S.(1:10)
    y = (x[1] ∪ x[3]) ∪ (x[3] ∪ ((x[5] ∪ x[7]) ∪ x[1]))

    @test length(string(x)) > 0
    @test eltype(y) == Int
    @test length(y) == 4
    @test sort(collect(y)) == [1, 3, 5, 7]
    @test sort(collect(copy(y))) == [1, 3, 5, 7]
    @test length(collect(product(y, y))) == 16
end
