# How SparseConnectivityTracer works

!!! warning "Internals may change"
    The developer documentation refers to internals that may change without warning in a future release of SparseConnectivityTracer.
    Anything written on this page should be treated as if it was undocumented.
    Only functionality that is exported or part of the [user documentation](@ref api) adheres to semantic versioning.


SparseConnectivityTracer works by pushing `Real` number types called tracers through generic functions.
Currently, two tracer types are provided:

* [`GradientTracer`](@ref SparseConnectivityTracer.GradientTracer): used for Jacobian sparsity patterns
* [`HessianTracer`](@ref SparseConnectivityTracer.HessianTracer): used for Hessian sparsity patterns

When used alone, these tracers compute [**global** sparsity patterns](@ref TracerSparsityDetector).
Alternatively, these can be used inside of a dual number type [`Dual`](@ref SparseConnectivityTracer.Dual), 
which keeps track of the primal computation and allows tracing through comparisons and control flow.
This is how [**local** spasity patterns](@ref TracerLocalSparsityDetector) are computed.

!!! tip "Tip: SparseConnectivityTracer as binary ForwardDiff"
     SparseConnectivityTracer's `Dual{T, GradientTracer}` can be thought of as a binary version of [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)'s own `Dual` number type.
     This is a good mental model for SparseConnectivityTracer if you are already familiar with ForwardDiff and its limitations.


## Index Sets

Let's take a look at a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$.
The gradient is defined as the vector $\frac{\partial f}{\partial x_i}$ 
and the Hessian as the matrix $\frac{\partial^2 f}{\partial x_i \partial x_j}$ for a given input $x\in\mathbb{R}^n$.


## Operator overloading: Toy example