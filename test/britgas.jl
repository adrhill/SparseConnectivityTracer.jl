include("britgas_definition.jl")

x0 = rand(450);
sum_britgas_cons(x0);

using SparseConnectivityTracer
using ADTypes: jacobian_sparsity, hessian_sparsity

@time "first jacobian" jacobian_sparsity(britgas_cons, x0, TracerSparsityDetector());  # slow
@time "second jacobian" jacobian_sparsity(britgas_cons, x0, TracerSparsityDetector());  # fast

@time "first hessian" hessian_sparsity(sum_britgas_cons, x0, TracerSparsityDetector());  # slow
@time "second hessian" hessian_sparsity(sum_britgas_cons, x0, TracerSparsityDetector());  # fast
