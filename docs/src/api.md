
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
SparseConnectivityTracer works by pushing `Number` types called tracers (e.g. [`ConnectivityTracer`](@ref) or [`ConnectivityTracer`](@ref)) through generic functions:

```@docs
JacobianTracer
ConnectivityTracer
tracer
trace_input
```

The following utilities can be used to extract input indices from a [`JacobianTracer`](@ref) or [`ConnectivityTracer`](@ref):

```@docs
inputs
```
