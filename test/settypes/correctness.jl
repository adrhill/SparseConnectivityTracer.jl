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
    P = collect(product(y, y)) # (1,1), (1,3), (1,5), (1,7), (3,3), (3,5), (3,7), (5,5), (5,7), (7,7)
    if S <: Union{BitSet,Set}
        @test length(P) == 10
    else
        @test length(P) == 16
    end
end
