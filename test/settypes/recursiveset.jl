using SparseConnectivityTracer: RecursiveSet
using Test

x = RecursiveSet.(1:10)

rs = union(
    union(x[1], x[3]),  #
    union(  #
        x[3],  #
        union(  #
            union(x[5], x[7]),  #
            x[1],  #
        ),
    ),
)

string(rs)

@test sort(collect(rs)) == [1, 3, 5, 7]
