
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## ADTypes Interface

SparseConnectivityTracer uses [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s interface for [sparsity detection](https://sciml.github.io/ADTypes.jl/stable/#Sparsity-detector).
In fact, the functions `jacobian_sparsity` and `hessian_sparsity` are re-exported from ADTypes.

To compute **global** sparsity patterns of `f(x)` over the entire input domain `x`, use
```@docs
TracerSparsityDetector
```

To compute **local** sparsity patterns of `f(x)` at a specific input `x`, use
```@docs
TracerLocalSparsityDetector
```

## Internals

!!! warning
    Internals may change without warning in a future release of SparseConnectivityTracer.

SparseConnectivityTracer works by pushing `Real` number types called tracers through generic functions.
Currently, two tracer types are provided:

```@docs
SparseConnectivityTracer.GradientTracer
SparseConnectivityTracer.HessianTracer
```

These can be used alone or inside of the dual number type `Dual`,
which keeps track of the primal computation and allows tracing through comparisons and control flow:

```@docs
SparseConnectivityTracer.Dual
```
