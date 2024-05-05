using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: RecursiveSet, SortedVector
using Symbolics: Symbolics
using NNlib: conv

include("brusselator.jl")

function benchmark_brusselator(N::Integer, method=:tracer_bitset)
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

    if method == :tracer_bitset
        return @benchmark jacobian_pattern($f!, $du, $u, BitSet)
    elseif method == :tracer_recursiveset
        return @benchmark jacobian_pattern($f!, $du, $u, RecursiveSet{UInt64})
    elseif method == :tracer_sortedvector
        return @benchmark jacobian_pattern($f!, $du, $u, SortedVector{UInt64})
    elseif method == :symbolics
        return @benchmark Symbolics.jacobian_sparsity($f!, $du, $u)
    end
end

function benchmark_conv(N, method=:tracer_bitset)
    x = rand(N, N, 3, 1) # WHCN image 
    w = rand(5, 5, 3, 2)  # corresponds to Conv((5, 5), 3 => 2)
    f(x) = conv(x, w)

    if method == :tracer_bitset
        return @benchmark jacobian_pattern($f, $x, BitSet)
    elseif method == :tracer_sortedvector
        return @benchmark jacobian_pattern($f, $x, SortedVector{UInt64})
    elseif method == :symbolics
        return @benchmark Symbolics.jacobian_sparsity($f, $x)
    end
end

## Run Brusselator benchmarks
for N in (6, 24, 100)
    for method in (:tracer_bitset, :tracer_sortedvector, :symbolics)
        @info "Benchmarking Brusselator of size $N with $method..."
        b = benchmark_brusselator(N, method)
        display(b)
    end
end

## Run conv benchmarks
for N in (28, 224)
    for method in (:tracer_bitset, :tracer_sortedvector) # Symbolics fails on this example
        @info "Benchmarking NNlib.conv on image of size ($N, $N, 3) with with $method..."
        b = benchmark_conv(N, method)
        display(b)
    end
end
