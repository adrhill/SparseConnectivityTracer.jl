
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
SparseConnectivityTracer works by pushing a `Number` type called [`ConnectivityTracer`](@ref) through generic functions:
```@docs
ConnectivityTracer
tracer
trace_input
```

The following utilities can be used to extract input indices from [`ConnectivityTracer`](@ref)s:
```@docs
inputs
```
