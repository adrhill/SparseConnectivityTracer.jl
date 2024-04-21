
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## Interface
```@docs
connectivity
TracerSparsityDetector
```

## Internals
SparseConnectivityTracer works by pushing a `Number` type called [`Tracer`](@ref) through generic functions:
```@docs
Tracer
tracer
trace_input
```

The following utilities can be used to extract input indices from [`Tracer`](@ref)s:
```@docs
inputs
```
