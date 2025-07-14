# [Performance](@id performance)

## Data structures for sparsity pattern representations

The most efficient internal data structure for sparsity pattern representations
depends on the number of inputs and the computational graph / sparsity of a given function.

Let's use a convolutional layer from Flux.jl as an example.
By default, SCT uses `BitSet` for Jacobian sparsity detection, which is well suited for small to medium sized functions.

```@example Flux
using SparseConnectivityTracer, Flux, BenchmarkTools

x = rand(28, 28, 3, 1)
layer = Conv((3, 3), 3 => 2)

detector_bitset = TracerSparsityDetector()
jacobian_sparsity(layer, x, detector_bitset)
```

```@example Flux
@benchmark jacobian_sparsity(layer, x, detector_bitset);
```

Instead of `BitSet`, we can use any concrete subtype of `AbstractSet{<:Integer}`, for example `Set{UInt}`.
To set the sparsity pattern type for Jacobian sparsity detection, we use the keyword argument `gradient_pattern_type`:

```@example Flux
detector_set = TracerSparsityDetector(; gradient_pattern_type=Set{UInt})
@benchmark jacobian_sparsity(layer, x, detector_set);
```

While this is slower for the given input size, the performance is highly dependant on the problem.
For larger inputs (e.g. of size $224 \times 224 \times 3 \times 1$), `detector_set` will outperform `detector_bitset`.
Note that memory requirement will vary as well.

For Hessians sparsity detection, the internal sparsity pattern representation uses either concrete subtypes of
`AbstractDict{I, AbstractSet{I}}` or `AbstractSet{Tuple{I, I}}`, where `I <: Integer`.
By default, `Dict{Int, BitSet)` is used.
To set the sparsity pattern type, use the keyword argument `hessian_pattern_type`:

```@example Flux
detector = TracerSparsityDetector(; hessian_pattern_type=Dict{UInt, Set{UInt}})
```

Data structures can also be set analogously for `TracerLocalSparsityDetector`.
If both Jacobian and Hessian sparsity patterns are needed, 
`gradient_pattern_type` and `hessian_pattern_type` can be set separately.
