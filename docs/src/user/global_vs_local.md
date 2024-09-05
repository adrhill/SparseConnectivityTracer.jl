# [Global vs. Local Sparsity](@id global-vs-local)

Let's motivate the difference between local and global sparsity patterns by taking a look at the function $f(\mathbf{x}) = x_1x_2$. 
The corresponding Jacobian is:

```math
J_f = \begin{bmatrix}
    \frac{\partial f}{\partial x_1} &
    \frac{\partial f}{\partial x_2}
\end{bmatrix}
=
\begin{bmatrix}
    x_2 & x_1
\end{bmatrix}
```

Depending on the values of $\mathbf{x}$, the resulting **local** Jacobian sparsity pattern could be either:
*  $[1\; 1]$ for $x_1 \neq 0$, $x_2 \neq 0$
*  $[1\; 0]$ for $x_1 = 0$, $x_2 \neq 0$
*  $[0\; 1]$ for $x_1 \neq 0$, $x_2 = 0$
*  $[0\; 0]$ for $x_1 = 0$, $x_2 = 0$

These are computed by [`TracerLocalSparsityDetector`](@ref):

```@repl localvsglobal
using SparseConnectivityTracer
detector = TracerLocalSparsityDetector();

f(x) = x[1]*x[2];

jacobian_sparsity(f, [1, 1], detector)
jacobian_sparsity(f, [0, 1], detector)
jacobian_sparsity(f, [1, 0], detector)
jacobian_sparsity(f, [0, 0], detector)
```

In contrast to this, [`TracerSparsityDetector`](@ref) computes a conservative union over all sparsity patterns in $\mathbf{x} \in \mathbb{R}^2$.
The resulting **global** pattern therefore does not depend on the input.
All of the following function calls are equivalent:

```@repl localvsglobal
detector = TracerSparsityDetector()

jacobian_sparsity(f, [1, 1], detector)
jacobian_sparsity(f, [0, 1], detector)
jacobian_sparsity(f, [1, 0], detector)
jacobian_sparsity(f, [0, 0], detector)
jacobian_sparsity(f, rand(2), detector)
```

!!! tip "Global vs. Local"
    Global sparsity patterns are the union of all local sparsity patterns over the entire input domain.
    For a given function, they are therefore always supersets of local sparsity patterns 
    and more "conservative" in the sense that they are less sparse.
