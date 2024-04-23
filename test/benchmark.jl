using BenchmarkTools
using SparseConnectivityTracer
using Symbolics: Symbolics
using NNlib: conv

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
        return @benchmark pattern($f!, $du, $u)
    elseif method == :symbolics
        return @benchmark Symbolics.jacobian_sparsity($f!, $du, $u)
    end
end

function benchmark_conv(method=:tracer)
    x = rand(28, 28, 3, 1) # WHCN image 
    w = rand(5, 5, 3, 16)  # corresponds to Conv((5, 5), 3 => 16)
    f(x) = conv(x, w)

    if method == :tracer
        return @benchmark pattern($f, $x)
    elseif method == :symbolics
        return @benchmark Symbolics.jacobian_sparsity($f, $x)
    end
end

## Run Brusselator benchmarks
for N in (6, 24)
    for method in (:tracer, :symbolics)
        @info "Benchmarking Brusselator of size $N with $method..."
        b = benchmark_brusselator(N, method)
        display(b)
    end
end

## Run conv benchmarks
@info "Benchmarking NNlib.conv with tracer..."
# Symbolics fails on this example
b = benchmark_conv(:tracer)
display(b)
