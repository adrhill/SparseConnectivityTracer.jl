```@meta
CurrentModule = SparseConnectivityTracer
```

# SparseConnectivityTracer

Documentation for [SparseConnectivityTracer](https://github.com/adrhill/SparseConnectivityTracer.jl).

## API reference
```@index
```

### Interface
```@docs
connectivity
```

### Internals
SparseConnectivityTracer works by pushing a `Number` type called [`Tracer`](@ref) through generic functions:
```@docs
Tracer
tracer
trace_input
```

The following utilities can be used to extract input indices from [`Tracer`](@ref)s:
```@docs
inputs
sortedinputs
```
