
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## Interface
```@docs
connectivity_pattern
jacobian_pattern
hessian_pattern
```
```@docs
TracerSparsityDetector
```

## Internals
SparseConnectivityTracer works by pushing `Number` types called tracers through generic functions.
Currently, three tracer types are provided:

```@docs
ConnectivityTracer
GlobalGradientTracer
GlobalHessianTracer
```

We also define alternative pseudo-set types that can deliver faster `union`:

```@docs
SparseConnectivityTracer.DuplicateVector
SparseConnectivityTracer.RecursiveSet
SparseConnectivityTracer.SortedVector
```
