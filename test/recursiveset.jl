using AbstractTrees
using SparseConnectivityTracer: RecursiveSet

x = RecursiveSet.(1:10)

s = union(x[1], union(x[2], union(x[3], x[4])))
