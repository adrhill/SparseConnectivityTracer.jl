# [How SparseConnectivityTracer works](@id how-sct-works)

!!! tip "Read the paper"
    Please read our TMLR paper describing SparseConnectivityTracer:
    [*Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians*](https://openreview.net/forum?id=GtXSN52nIW).

!!! danger "Internals may change"
    The developer documentation might refer to internals which can change without warning in a future release of SparseConnectivityTracer.
    Only functionality that is exported or part of the [user documentation](@ref api) adheres to semantic versioning.

## Tracers are scalars

SparseConnectivityTracer (SCT) works by pushing `Real` number types called tracers through generic functions using operator overloading.
Currently, two tracer types are provided:

* [`GradientTracer`](@ref SparseConnectivityTracer.GradientTracer): used for Jacobian sparsity patterns
* [`HessianTracer`](@ref SparseConnectivityTracer.HessianTracer): used for Hessian sparsity patterns

When used alone, these tracers compute [**global** sparsity patterns](@ref TracerSparsityDetector).
Alternatively, these can be used inside of a dual number type [`Dual`](@ref SparseConnectivityTracer.Dual), 
which keeps track of the primal computation and allows tracing through comparisons and control flow.
This is how [**local** sparsity patterns](@ref TracerLocalSparsityDetector) are computed.

!!! tip "Tip: View SparseConnectivityTracer as binary ForwardDiff"
     SparseConnectivityTracer's `Dual{T, GradientTracer}` can be thought of as a binary version of [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)'s own `Dual` number type.
     This is a good mental model for SparseConnectivityTracer if you are familiar with ForwardDiff and its limitations.

## Index sets

Let's take a look at a scalar function $f: \mathbb{R}^n \rightarrow \mathbb{R}$.
For a given input $\mathbf{x} \in \mathbb{R}^n$, 
the gradient of $f$ is defined as $\left(\nabla f(\mathbf{x})\right)_{i} = \frac{\partial f}{\partial x_i}$ 
and the Hessian as $\left(\nabla^2 f(\mathbf{x})\right)_{i,j} = \frac{\partial^2 f}{\partial x_i \partial x_j}$. 

Sparsity patterns correspond to the mask of non-zero values in the gradient and Hessian.
Instead of saving the values of individual partial derivatives, they can efficiently be represented by the set of indices corresponding to non-zero values:

* Gradient patterns are represented by sets of indices $\left\{i \;\big|\; \frac{\partial f}{\partial x_i} \neq 0\right\}$
* Hessian patterns are represented by sets of index tuples $\left\{(i, j) \;\Big|\; \frac{\partial^2 f}{\partial x_i \partial x_j} \neq 0\right\}$

!!! warning "Global vs. Local"
    As shown in the page [*"Global vs. Local"*](@ref global-vs-local),
    global sparsity patterns are the index sets over all $\mathbf{x}\in\mathbb{R}^n$,
    whereas local patterns are the index sets for a given point $\mathbf{x}$.
    For a given function $f$, global sparsity patterns are therefore always supersets of local sparsity patterns 
    and more "conservative" in the sense that they are less sparse. 

## Motivating example

Let's take a look at the computational graph of the equation $f(\mathbf{x}) = x_1 + x_2x_3 + \text{sgn}(x_4)$,
where $\text{sgn}$ is the [sign function](https://en.wikipedia.org/wiki/Sign_function):

```mermaid
flowchart LR
    subgraph Inputs
    X1["$$x_1$$"]
    X2["$$x_2$$"]
    X3["$$x_3$$"]
    X4["$$x_4$$"]
    end

    PLUS((+))
    TIMES((*))
    SIGN((sgn))
    PLUS2((+))

    X1 --> |"{1}"| PLUS
    X2 --> |"{2}"| TIMES
    X3 --> |"{3}"| TIMES
    X4 --> |"{4}"| SIGN
    TIMES  --> |"{2,3}"| PLUS
    PLUS --> |"{1,2,3}"| PLUS2
    SIGN --> |"{}"| PLUS2

    PLUS2 --> |"{1,2,3}"| RES["$$y=f(x)$$"]
```
To obtain a sparsity pattern, each scalar input $x_i$ gets seeded with a corresponding singleton index set $\{i\}$ [^1]. 
Since addition and multiplication have non-zero derivatives with respect to both of their inputs, 
their outputs accumulate and propagate the index sets of their inputs (annotated on the edges of the graph above).
The sign function has zero derivatives for any input value. It therefore doesn't propagate the index set ${4}$ corresponding to the input $x_4$. Instead, it returns an empty set.

[^1]: $\frac{\partial x_i}{\partial x_j} \neq 0$ only holds for $i=j$

The resulting **global** gradient sparsity pattern $\left(\nabla f(\mathbf{x})\right)_{i} \neq 0$ for $i$ in $\{1, 2, 3\}$ matches the analytical gradient

```math 
\nabla f(\mathbf{x}) = \begin{bmatrix}
    \frac{\partial f}{\partial x_1} \\
    \frac{\partial f}{\partial x_2} \\
    \frac{\partial f}{\partial x_3} \\
    \frac{\partial f}{\partial x_4}
\end{bmatrix}
=
\begin{bmatrix}
    1 \\
    x_3 \\
    x_2 \\
    0
\end{bmatrix} \quad .
```

!!! tip "From Global to Local"
    Note that the **local** sparsity pattern could be more sparse in case $x_2$ and/or $x_3$ are zero.
    Computing such local sparsity patterns requires [`Dual`](@ref SparseConnectivityTracer.Dual) numbers with information about the primal computation. 
    These are used to evaluate the **local** differentiability of operations like multiplication.

## Toy implementation

As mentioned above, SCT uses operator overloading to keep track of index sets.
Let's start by implementing our own `MyGradientTracer` type:

```@example toytracer
struct MyGradientTracer
    indexset::Set
end
```

We can now overload operators from Julia Base using our type:

```@example toytracer
import Base: +, *, sign

Base.:+(a::MyGradientTracer, b::MyGradientTracer) = MyGradientTracer(union(a.indexset, b.indexset))
Base.:*(a::MyGradientTracer, b::MyGradientTracer) = MyGradientTracer(union(a.indexset, b.indexset))
Base.sign(x::MyGradientTracer) = MyGradientTracer(Set()) # return empty index set
```

Let's create a vector of tracers to represent our input and evaluate our function with it:

```@example toytracer
f(x) = x[1] + x[2]*x[3] * sign(x[4])

xtracer = [
    MyGradientTracer(Set(1)),
    MyGradientTracer(Set(2)),
    MyGradientTracer(Set(3)),
    MyGradientTracer(Set(4)),
]

ytracer = f(xtracer)
```

Compared to this toy implementation, SCT adds some utilities to automatically create `xtracer` and parse the output `ytracer` into a sparse matrix, which we will omit here.

[`jacobian_sparsity(f, x, TracerSparsityDetector())`](@ref TracerSparsityDetector) calls these three steps of (1) tracer creation, (2) function evaluation and (3) output parsing in sequence:

```@example toytracer
using SparseConnectivityTracer

x = rand(4)
jacobian_sparsity(f, x, TracerSparsityDetector())
```

## Tracing Jacobians

Our toy implementation above doesn't just work on scalar functions, but also on vector valued functions:

```@example toytracer
g(x) = [x[1], x[2]*x[3], x[1]+x[4]]
g(xtracer)
```

By stacking individual `MyGradientTracer`s row-wise, we obtain the sparsity pattern of the Jacobian of $g$

```math
J_g(\mathbf{x})=
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & x_3 & x_2 & 0 \\
1 & 0 & 0 & 1
\end{pmatrix} \quad .
```

We obtain the same result using SCT's `jacobian_sparsity`:
```@example toytracer
jacobian_sparsity(g, x, TracerSparsityDetector())
```

## Tracing Hessians

In the sections above, we outlined how to implement our own [`GradientTracer`](@ref SparseConnectivityTracer.GradientTracer) from scratch.
[`HessianTracer`](@ref SparseConnectivityTracer.HessianTracer) use the same operator overloading approach but are a bit more involved as they contain two index sets: 
one for the gradient pattern and one for the Hessian pattern. 
These sets are updated based on whether the first- and second-order derivatives of an operator are zero or not.

!!! tip "To be published"
    Look forward to our upcoming publication of SparseConnectivityTracer, 
    where we will go into more detail on the implementation of `HessianTracer`!
