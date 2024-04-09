using BenchmarkTools
using SparseConnectivityTracer
using Symbolics: Symbolics

include("brusselator.jl")

function benchmark_brusselator(N::Integer, method=:tracer)
    dims = (N, N, 2)
    A = 1.0
    B = 1.0
    alpha = 1.0
    xyd = fill(1.0, N)
    dx = 1.0
    p = (A, B, alpha, xyd, dx, N)

    u = rand(dims...)
    du = similar(u)
    f!(du, u) = brusselator_2d_loop(du, u, p, nothing)

    if method == :tracer
        return @benchmark connectivity($f!, $du, $u)
    elseif method == :symbolics
        return @benchmark Symbolics.jacobian_sparsity($f!, $du, $u)
    end
end

benchmark_brusselator(6, :tracer)
benchmark_brusselator(6, :symbolics)

benchmark_brusselator(24, :tracer)
benchmark_brusselator(24, :symbolics)
