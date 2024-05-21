
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## ADTypes Interface

For package developers, we recommend using the [ADTypes.jl](https://github.com/SciML/ADTypes.jl) interface.

To compute global sparsity patterns of `f(x)` over the entire input domain `x`, use
```@docs
TracerSparsityDetector
```

To compute local sparsity patterns of `f(x)` at a specific input `x`, use
```@docs
TracerLocalSparsityDetector
```

## Legacy Interface

### Global sparsity 

The following functions can be used to compute global sparsity patterns of `f(x)` over the entire input domain `x`.

```@docs
connectivity_pattern
jacobian_pattern
hessian_pattern
```

[`TracerSparsityDetector`](@ref) is the ADTypes equivalent of these functions.

### Local sparsity

The following functions can be used to compute local sparsity patterns of `f(x)` at a specific input `x`.
Note that these patterns are sparser than global patterns but need to be recomputed when `x` changes.

```@docs
local_connectivity_pattern
local_jacobian_pattern
local_hessian_pattern
```

[`TracerLocalSparsityDetector`](@ref) is the ADTypes equivalent of these functions.

## Internals

!!! warning
    Internals may change without warning in a future release of SparseConnectivityTracer.

SparseConnectivityTracer works by pushing `Number` types called tracers through generic functions.
Currently, three tracer types are provided:

```@docs
SparseConnectivityTracer.ConnectivityTracer
SparseConnectivityTracer.GradientTracer
SparseConnectivityTracer.HessianTracer
```

These can be used alone or inside of the dual number type `Dual`,
which keeps track of the primal computation and allows tracing through comparisons and control flow:

```@docs
SparseConnectivityTracer.Dual
```

We also define alternative pseudo-set types that can deliver faster `union`:

```@docs
SparseConnectivityTracer.DuplicateVector
SparseConnectivityTracer.RecursiveSet
SparseConnectivityTracer.SortedVector
```
