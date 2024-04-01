```@meta
CurrentModule = SparseConnectivityTracer
```

# SparseConnectivityTracer

Documentation for [SparseConnectivityTracer](https://github.com/adrhill/SparseConnectivityTracer.jl).

```@index
```

## API reference
SparseConnectivityTracer works by pushing a `Number` type called [`Tracer`](@ref) through generic functions:
```@docs
Tracer
tracer
```

The resulting connectivity matrix can be extracted using [`connectivity`](@ref): 
```@docs
connectivity
```

or manually from individual [`Tracer`](@ref) outputs:
```@docs
inputs
sortedinputs
```
