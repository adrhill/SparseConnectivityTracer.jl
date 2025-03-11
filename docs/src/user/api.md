
```@meta
CurrentModule = Main
CollapsedDocStrings = true
```

# [API Reference](@id api)

SparseConnectivityTracer uses [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s [interface for sparsity detection](https://sciml.github.io/ADTypes.jl/stable/#Sparsity-detector).
In fact, the functions `jacobian_sparsity` and `hessian_sparsity` are re-exported from ADTypes.

```@docs
ADTypes.jacobian_sparsity
ADTypes.hessian_sparsity
```

To compute **global** sparsity patterns of `f(x)` over the entire input domain `x`, use
```@docs
TracerSparsityDetector
```

To compute **local** sparsity patterns of `f(x)` at a specific input `x`, use
```@docs
TracerLocalSparsityDetector
```

## Memory allocation

For developers requiring the allocation of output buffers that support our tracers, we additionally provide
```@docs
jacobian_buffer
hessian_buffer
```
