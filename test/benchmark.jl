using ADTypes: AbstractSparsityDetector, jacobian_sparsity
using BenchmarkTools
using DifferentiationInterface: SymbolicsSparsityDetector
using SparseConnectivityTracer
using SparseConnectivityTracer: DuplicateVector, RecursiveSet, SortedVector
using Symbolics: Symbolics
using NNlib: conv

all_sparsity_detectors = [
    SymbolicsSparsityDetector(),
    TracerSparsityDetector(BitSet),
    TracerSparsityDetector(Set{UInt64}),
    TracerSparsityDetector(DuplicateVector{UInt64}),
    TracerSparsityDetector(RecursiveSet{UInt64}),
    TracerSparsityDetector(SortedVector{UInt64}),
]

include("brusselator_definition.jl")

function benchmark_brusselator(N::Integer, sparsity_detector::AbstractSparsityDetector)
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

    return @btime jacobian_sparsity($f!, $du, $u, $sparsity_detector)
end

function benchmark_conv(N::Integer, sparsity_detector::AbstractSparsityDetector)
    x = rand(N, N, 3, 1) # WHCN image 
    w = rand(5, 5, 3, 2)  # corresponds to Conv((5, 5), 3 => 2)
    f(x) = conv(x, w)
    return @btime jacobian_sparsity($f, $x, $sparsity_detector)
end

## Run Brusselator benchmarks
for N in (6, 24, 100), sparsity_detector in all_sparsity_detectors
    @info "Benchmarking Brusselator of size $N with $sparsity_detector"
    benchmark_brusselator(N, sparsity_detector)
end

## Run conv benchmarks
for N in (28, 224), sparsity_detector in all_sparsity_detectors[2:end]
    # Symbolics fails on this example
    @info "Benchmarking NNlib.conv on image of size ($N, $N, 3) with $sparsity_detector"
    benchmark_conv(N, sparsity_detector)
end
