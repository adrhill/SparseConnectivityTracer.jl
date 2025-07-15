# [Internals Reference](@id internal-api)

!!! danger "Internals may change"
    This part of the developer documentation **exclusively** refers to internals that may change without warning in a future release of SparseConnectivityTracer.
    Anything written on this page should be treated as if it was undocumented.
    Only functionality that is exported or part of the [user documentation](@ref api) adheres to semantic versioning.


```@index
```

## Tracer Types

```@docs
SparseConnectivityTracer.AbstractTracer
SparseConnectivityTracer.GradientTracer
SparseConnectivityTracer.HessianTracer
SparseConnectivityTracer.Dual
```

### Traits

```@docs
SparseConnectivityTracer.NotShared
SparseConnectivityTracer.Shared
SparseConnectivityTracer.shared
```

### Utilities

```@docs
SparseConnectivityTracer.gradient
SparseConnectivityTracer.hessian
SparseConnectivityTracer.myempty
```