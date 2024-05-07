using ADTypes: AbstractSparsityDetector, jacobian_sparsity
using BenchmarkTools
using SparseConnectivityTracer
using SparseConnectivityTracer: SortedVector
using NNlib: conv

include("../test/brusselator_definition.jl")

const METHODS = (
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(DuplicateVector{UInt64}),
    TracerSparsityDetector(RecursiveSet{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
)

function benchmark_brusselator(N::Integer, method::AbstractSparsityDetector)
    f! = Brusselator!(N)
    x = rand(N, N, 2)
    y = similar(x)

    return @benchmark jacobian_sparsity($f!, $y, $x, $method)
end

function benchmark_conv(N, method::AbstractSparsityDetector)
    x = rand(N, N, 3, 1) # WHCN image 
    w = rand(5, 5, 3, 2)  # corresponds to Conv((5, 5), 3 => 2)
    f(x) = conv(x, w)

    return @benchmark jacobian_sparsity($f, $x, $method)
end

## Run Brusselator benchmarks
for N in (6, 24, 100)
    for method in METHODS
        @info "Benchmarking Brusselator of size $N with $method..."
        b = benchmark_brusselator(N, method)
        display(b)
    end
end

## Run conv benchmarks
for N in (28, 128)
    for method in METHODS # Symbolics fails on this example
        @info "Benchmarking NNlib.conv on image of size ($N, $N, 3) with with $method..."
        b = benchmark_conv(N, method)
        display(b)
    end
end
