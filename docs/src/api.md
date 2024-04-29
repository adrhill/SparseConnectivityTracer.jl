
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# API Reference
```@index
```

## Interface
```@docs
pattern
TracerSparsityDetector
```

## Internals
SparseConnectivityTracer works by pushing `Number` types called tracers through generic functions.
Currently, two tracer types are provided:

```@docs
ConnectivityTracer
JacobianTracer
HessianTracer
```

Utilities to create tracers:
```@docs
tracer
trace_input
```

Utility to extract input indices from tracers:
```@docs
inputs
```
