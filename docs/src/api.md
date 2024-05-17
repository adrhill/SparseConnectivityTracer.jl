
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## Interface

### Global sparsity 

The following functions can be used to compute global sparsity patterns of `f(x)` over the entire input domain `x`.

```@docs
connectivity_pattern
jacobian_pattern
hessian_pattern
```

Alternatively, [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s interface can be used:
```@docs
TracerSparsityDetector
```

### Local sparsity

The following functions can be used to compute local sparsity patterns of `f(x)` at a specific input `x`.
Note that these patterns are sparser than global patterns but need to be recomputed when `x` changes.

```@docs
local_connectivity_pattern
local_jacobian_pattern
local_hessian_pattern
```

Note that [ADTypes.jl](https://github.com/SciML/ADTypes.jl) doesn't provide an interface for local sparsity detection.

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

These can be used alone or inside of the dual number type [`Dual`](@ref),
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
