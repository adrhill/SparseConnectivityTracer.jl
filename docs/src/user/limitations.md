# [Limitations](@id limitations)

## Sparsity patterns are conservative approximations

Sparsity patterns returned by SparseConnectivityTracer (SCT) can in some cases be overly conservative, meaning that they might contain "too many ones".
If you observe an overly conservative pattern, [please open a feature request](https://github.com/adrhill/SparseConnectivityTracer.jl/issues) so we know where to add more method overloads to increase the sparsity.

!!! warning "SCT's no-false-negatives policy"
    If you ever observe a sparsity pattern that contains too many zeros, we urge you to [open a bug report](https://github.com/adrhill/SparseConnectivityTracer.jl/issues)!

## Function must be composed of generic Julia functions

SCT can't trace through non-Julia code.
However, if you know the sparsity pattern of an external, non-Julia function,
you might be able to work around it by adding methods on SCT's tracer types.

## Function types must be generic

When computing the sparsity pattern of a function,
it must be written generically enough to accept numbers of type `T<:Real` as (or `AbstractArray{<:Real}`) as inputs.

!!! details "Example: Overly restrictive type annotations"
    Let's see this mistake in action:

    ```@example notgeneric
    using SparseConnectivityTracer
    detector = TracerSparsityDetector()

    relu_bad(x::AbstractFloat) = max(zero(x), x)

    f_bad(xs) = sum(relu_bad, xs)
    nothing # hide
    ```

    Since tracers and dual numbers are `Real` numbers and not `AbstractFloat`s,
    `relu_bad` throws a `MethodError`:

    ```@repl notgeneric
    xs = [1.0, -2.0, 3.0];

    f_bad(xs)

    jacobian_sparsity(f_bad, xs, detector)
    ```

    This is easily fixed by loosening type restrictions or adding an additional methods on `Real`:

    ```@example notgeneric
    relu_good(x) = max(zero(x), x)

    f_good(xs) = sum(relu_good, xs)
    nothing # hide
    ```

    ```@repl notgeneric
    jacobian_sparsity(f_good, xs, detector)
    ```

## Limited control flow

Only [`TracerLocalSparsityDetector`](@ref) supports comparison operators (`<`, `==`, ...), indicator functions (`iszero`, `iseven`, ...) and control flow.

[`TracerSparsityDetector`](@ref) does not support any boolean functions and control flow (with the exception of `ifelse`).
This might seem unintuitive but follows from our policy stated above: SCT guarantees conservative sparsity patterns.
Using an approach based on operator-overloading, this means that global sparsity detection isn't allowed to hit any branching code.
`ifelse` is the only exception, since it allows us to evaluate both branches.


!!! warning "Common control flow errors"
    By design, SCT will throw errors instead of returning wrong sparsity patterns. Common error messages include:

    ```julia
    ERROR: TypeError: non-boolean [tracer type] used in boolean context
    ```
    
    ```julia
    ERROR: Function [function] requires primal value(s).
    A dual-number tracer for local sparsity detection can be used via `TracerLocalSparsityDetector`.
    ```

!!! details "Why does TracerSparsityDetector not support control flow and comparisons?"
    Let us motivate the design decision above by a simple example function:

    ```@example ctrlflow
    function f(x)
        if x[1] > x[2]
            return x[1]
        else 
            return x[2]
        end
    end
    nothing # hide
    ```

    The desired **global** Jacobian sparsity pattern over the entire input domain $x \in \mathbb{R}^2$ is `[1 1]`. 
    Two **local** sparsity patterns are possible: 
    `[1 0]` for $\{x | x_1 > x_2\}$,
    `[0 1]` for $\{x | x_1 \le x_2\}$.

    The local sparsity patterns of [`TracerLocalSparsityDetector`](@ref) are easy to compute using operator overloading by using [dual numbers](@ref SparseConnectivityTracer.Dual) 
    which contain primal values on which we can evaluate comparisons like `>`:

    ```@example ctrlflow
    using SparseConnectivityTracer

    jacobian_sparsity(f, [2, 1], TracerLocalSparsityDetector())

    jacobian_sparsity(f, [1, 2], TracerLocalSparsityDetector())
    ```

    The global sparsity pattern is **impossible** to compute when code branches with an if-else condition, 
    since we can only ever hit one branch during run-time. 
    If we made comparisons like `>` return `true` or `false`, we'd get the local patterns `[1 0]` and `[0 1]` respectively. 
    But SCT's policy is to guarantee conservative sparsity patterns, which means that "false positives" (ones) are acceptable, but "false negatives" (zeros) are not.
    In my our opinion, the right thing to do here is to throw an error:

    ```@repl ctrlflow
    jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

    In some cases, we can work around this by using `ifelse`.
    Since `ifelse` is a method, it can evaluate "both branches" and take a conservative union of both resulting sparsity patterns:

    ```@example ctrlflow
    f(x) = ifelse(x[1] > x[2], x[1], x[2])

    jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

## [No guarantee of conservative global sparsity patterns on stateful code](@id stateful-code)

SCT can't guarantee correct, conservative global sparsity patterns on stateful functions `f(x)` whose output isn't fully determined by the input `x`.
We provide some common examples:

!!! details "Example: Stateful branching code"

    As motivated in the section above, global sparsity detection isn't allowed to hit any branching code.
    While SCT's overloads try to avoid branches by throwing errors, they can in some cases be entered by stateful functions.
    Let's look at a function whose output doesn't only depend on the input `x`, but also on an internal state (in this case a random number):

    ```@repl stateful
    using SparseConnectivityTracer

    f(x) = randn() > 0.5 ? x[1] : x[2];

    jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

    SCT cannot return the correct global sparsity pattern `[1 1]` for this code.
    This issue can be circumvented by [adding an overload](@ref adding-overloads) on `f` that returns a conservative pattern.

!!! details "Example: Stateful mutable caches"

    Problems can also arise when statefulness is introduced via mutable caches.
    In this example, we assume a stateful function `f` whose output not only depends on the input `x`, but also on a mutable array defined outside of the function call:

    ```@repl stateful2
    using SparseConnectivityTracer, SparseArrays

    A_cache = sparse([2 0; 0 3])

    f(x) = A_cache * x;

    pattern1 = jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

    #### Invalidation of global sparsity patterns 

    While this sparsity pattern is correct at the time of detection,
    it can be invalidated by mutating the array `A_cache`:

    ```@repl stateful2
    A_cache[1, 2] = 4;

    A_cache

    pattern2 = jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

    With the wisdom of hindsight, `pattern1` can therefore be seen as "non-conservative".

    #### Exception: Dense arrays

    For dense caches of type `Array` (including `Matrix` and `Vector`), SCT tries to circumvent this issue by returning a conservative sparsity pattern
    *(note that this behavior could be changed in a future breaking release)*:

    ```@repl stateful2
    A_cache = [2 0; 0 3]

    pattern1 = jacobian_sparsity(f, [1, 2], TracerSparsityDetector())
    ```

    However, such guarantees can't be made for arbitrary cache types (like the `SparseMatrixCSC` above).
    