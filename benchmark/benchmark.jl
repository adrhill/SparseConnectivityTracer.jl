using BenchmarkTools

using ADTypes: AbstractSparsityDetector, jacobian_sparsity
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector

using SparseArrays: sprand
using SimpleDiffEq: ODEProblem, solve, SimpleEuler
using Flux: Conv

const SET_TYPES = (BitSet, Set{UInt64}, DuplicateVector{UInt64}, SortedVector{UInt64})

## ODEs
include("../test/brusselator_definition.jl")

## Iterated multiplication by random sparse matrices
function iterated_sparse_matmul(x; sparsity=0.1, iterations=5)
    n = length(x)
    y = copy(x)
    for _ in 1:iterations
        A = sprand(n, n, sparsity)
        y = A * y
    end
    return y
end

## Deep Learning
const LAYER = Conv((5, 5), 3 => 2)

## Define Benchmark suite
suite = BenchmarkGroup()

for S in SET_TYPES
    setname = string(S)
    method = TracerSparsityDetector(S)

    J_global = suite["Jacobian"]["Global"]

    ## Sparse matrix multiplication
    for sparsity in (0.01, 0.05, 0.1, 0.25, 0.5)
        x = rand(50)
        f(x) = iterated_sparse_matmul(x; sparsity=sparsity, iterations=5)
        J_global["sparse_matmul"]["sparsity=$sparsity"][setname] = @benchmarkable jacobian_sparsity(
            $f, $x, $method
        )
    end

    ## ODEs
    for N in (6, 24, 100)
        f! = Brusselator!(N)
        x = rand(N, N, 2)
        y = similar(x)
        J_global["brusselator"]["N=$N"][setname] = @benchmarkable jacobian_sparsity(
            $f!, $y, $x, $method
        )

        # TODO: test adaptive step solvers on local tracers
        solver = SimpleEuler()
        prob = ODEProblem(brusselator_2d_loop!, x, (0.0, 1.0), f!.params)
        function brusselator_ode_solve(x)
            return solve(ODEProblem(brusselator_2d_loop!, x, (0.0, 1.0), f!.params), solver; dt=0.2).u[end]
        end
        J_global["brusselator_ode_solve"]["N=$N"][setname] = @benchmarkable jacobian_sparsity(
            $brusselator_ode_solve, $x, $method
        )
    end

    ## Deep Learning
    for N in (28, 128)
        J_global["conv"]["size=$(N)x$(N)x3x1"][setname] = @benchmarkable jacobian_sparsity(
            $LAYER, $(rand(N, N, 3, 1)), $method
        )
    end
    # TODO: benchmark local sparsity tracers on LeNet-5 CNN
end
