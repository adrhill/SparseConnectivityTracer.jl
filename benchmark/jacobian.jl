using BenchmarkTools

using ADTypes: AbstractSparsityDetector, jacobian_sparsity
using SparseConnectivityTracer
using SparseConnectivityTracerBenchmarks.ODE: brusselator_2d_loop!

using SparseArrays: sprand
using SimpleDiffEq: ODEProblem, solve, SimpleEuler
using Flux: Conv

function jacbench(method)
    suite = BenchmarkGroup()
    suite["SparseMul"] = jacbench_sparsemul(method)
    suite["Brusselator"] = jacbench_brusselator(method)
    suite["Conv"] = jacbench_conv(method)
    return suite
end

## Iterated sparse mul

struct IteratedSparseMul{M<:AbstractMatrix}
    As::Vector{M}
end

function IteratedSparseMul(; n::Integer, p::Real=0.1, depth::Integer=5)
    As = [sprand(n, n, p) for _ in 1:depth]
    return IteratedSparseMul(As)
end

function (ism::IteratedSparseMul)(x::AbstractVector)
    @assert length(x) == size(ism.As[1], 1)
    y = copy(x)
    for l in eachindex(ism.As)
        y = ism.As[l] * y
    end
    return y
end

function jacbench_sparsemul(method)
    suite = BenchmarkGroup()
    for n in [50], p in [0.01, 0.25], depth in [5]
        x = rand(n)
        f = IteratedSparseMul(; n, p, depth)
        suite["(n=$n, p=$p, depth=$depth)"] = @benchmarkable jacobian_sparsity(
            $f, $x, $method
        )
    end
    return suite
end

## Brusselator

function jacbench_brusselator(method)
    suite = BenchmarkGroup()
    for N in (6, 24)
        f! = Brusselator!(N)
        x = rand(N, N, 2)
        y = similar(x)
        suite["operator"]["N=$N"] = @benchmarkable jacobian_sparsity($f!, $y, $x, $method)
        solver = SimpleEuler()
        prob = ODEProblem(brusselator_2d_loop!, x, (0.0, 1.0), f!.params)
        function brusselator_ode_solve(x)
            return solve(ODEProblem(brusselator_2d_loop!, x, (0.0, 1.0), f!.params), solver; dt=0.5).u[end]
        end
        suite["ODE"]["N=$N"] = @benchmarkable jacobian_sparsity(
            $brusselator_ode_solve, $x, $method
        )
    end
    return suite
end

## Convolution

function jacbench_conv(method)
    # TODO: benchmark local sparsity tracers on LeNet-5 CNN
    layer = Conv((5, 5), 3 => 2)
    suite = BenchmarkGroup()
    for N in (28, 128)
        suite["size=$(N)x$(N)x3"] = @benchmarkable jacobian_sparsity(
            $layer, $(rand(N, N, 3, 1)), $method
        )
    end
    return suite
end
