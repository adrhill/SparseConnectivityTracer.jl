using SparseConnectivityTracer: DuplicateVector
using Test

x = DuplicateVector{Int}.(1:10)

y = union(
    union(x[1], x[3]),  #
    union(  #
        x[3],  #
        union(  #
            union(x[5], x[7]),  #
            x[1],  #
        ),
    ),
)

@test sort(collect(y)) == [1, 3, 5, 7]
