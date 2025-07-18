# [Adding Overloads](@id adding-overloads)

!!! danger "Internals may change"
    The developer documentation might refer to internals which can change without warning in a future release of SparseConnectivityTracer.
    Only functionality that is exported or part of the [user documentation](@ref api) adheres to semantic versioning.

Having read our guide [*"How SparseConnectivityTracer works"*](@ref how-sct-works), you might want to add your own methods on 
[`GradientTracer`](@ref SparseConnectivityTracer.GradientTracer), 
[`HessianTracer`](@ref SparseConnectivityTracer.HessianTracer) and
[`Dual`](@ref SparseConnectivityTracer.Dual)
to improve the performance of your functions or to work around some of SCT's [limitations](@ref limitations).

!!! warning "Don't overload manually"
    If you want to overload a `Function` that takes `Real` arguments, 
    we strongly discourage you from manually adding methods to your function that use our internal tracer types.

    Instead, use the same code generation mechanisms that we use. This page shows you how.

## Generated overloads

!!! tip "Copy one of our package extensions"
    The easiest way to add overloads is to copy one of our package extensions, [e.g. our NNlib extension](https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/ext/SparseConnectivityTracerNNlibExt.jl), and to modify it.
    Please upstream your additions by opening a pull request! We will help you out to get your feature merged.

### Operator classification

SCT currently supports three types of functions:

1. **1-to-1**: operators with one input and one output
2. **2-to-1**: operators with two inputs and one output
3. **1-to-2**: operators with one input and two outputs

Depending on the type of function you're dealing with, you will have to specify the way in which your function is differentiable:

| In | Out | Examples                 | Methods you need to implement                                                                                                              | 
|:--:|:---:|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | 1   | `sin`, `cos`, `abs`      | `is_der1_zero_global`, `is_der2_zero_global`                                                                                               |
| 2  | 1   | `+`, `*`, `>`, `isequal` | `is_der1_arg1_zero_global`, `is_der2_arg1_zero_global`, `is_der1_arg2_zero_global`, `is_der2_arg2_zero_global`, `is_der_cross_zero_global` |
| 1  | 2   | `sincos`                 | `is_der1_out1_zero_global`, `is_der2_out1_zero_global`, `is_der1_out2_zero_global`, `is_der2_out2_zero_global`                             |


!!! details "Methods you have to implement for 1-to-1 operators"

    | Function                                   | Meaning                                                 |
    |:-------------------------------------------|:--------------------------------------------------------|
    | `is_der1_zero_global(::typeof(f)) = false` | $\frac{\partial f}{\partial x} \neq 0$ for some $x$     | 
    | `is_der2_zero_global(::typeof(f)) = false` | $\frac{\partial^2 f}{\partial x^2} \neq 0$ for some $x$ | 

    Optionally, to increase the sparsity of [`TracerLocalSparsityDetector`](@ref), you can additionally implement

    | Function                                     | Meaning                                                  |
    |:---------------------------------------------|:---------------------------------------------------------|
    | `is_der1_zero_local(::typeof(f), x) = false` | $\frac{\partial f}{\partial x} \neq 0$ for given $x$     | 
    | `is_der2_zero_local(::typeof(f), x) = false` | $\frac{\partial^2 f}{\partial x^2} \neq 0$ for given $x$ | 

    These fall back to 

    ```julia
    is_der1_zero_local(f::F, x) where {F} = is_der1_zero_global(f)
    is_der2_zero_local(f::F, x) where {F} = is_der2_zero_global(f)
    ```

!!! details "Methods you have to implement for 2-to-1 operators"

    | Function                                        | Meaning                                                            |
    |:------------------------------------------------|:-------------------------------------------------------------------|
    | `is_der1_arg1_zero_global(::typeof(f)) = false` | $\frac{\partial f}{\partial x} \neq 0$ for some $x,y$              | 
    | `is_der2_arg1_zero_global(::typeof(f)) = false` | $\frac{\partial^2 f}{\partial x^2} \neq 0$ for some $x,y$          | 
    | `is_der1_arg2_zero_global(::typeof(f)) = false` | $\frac{\partial f}{\partial y} \neq 0$ for some $x,y$              | 
    | `is_der2_arg2_zero_global(::typeof(f)) = false` | $\frac{\partial^2 f}{\partial y^2} \neq 0$ for some $x,y$          | 
    | `is_der_cross_zero_global(::typeof(f)) = false` | $\frac{\partial^2 f}{\partial x \partial y} \neq 0$ for some $x,y$ | 

    Optionally, to increase the sparsity of [`TracerLocalSparsityDetector`](@ref), you can additionally implement

    | Function                                             | Meaning                                                             |
    |:-----------------------------------------------------|:--------------------------------------------------------------------|
    | `is_der1_arg1_zero_local(::typeof(f), x, y) = false` | $\frac{\partial f}{\partial x} \neq 0$ for given $x,y$              | 
    | `is_der2_arg1_zero_local(::typeof(f), x, y) = false` | $\frac{\partial^2 f}{\partial x^2} \neq 0$ for given $x,y$          | 
    | `is_der1_arg2_zero_local(::typeof(f), x, y) = false` | $\frac{\partial f}{\partial x} \neq 0$ for given $x,y$              | 
    | `is_der2_arg2_zero_local(::typeof(f), x, y) = false` | $\frac{\partial^2 f}{\partial x^2} \neq 0$ for given $x,y$          | 
    | `is_der_cross_zero_local(::typeof(f), x, y) = false` | $\frac{\partial^2 f}{\partial x \partial y} \neq 0$ for given $x,y$ | 


    These fall back to 

    ```julia
    is_der1_arg1_zero_local(f::F, x, y) where {F} = is_der1_arg1_zero_global(f)
    is_der2_arg1_zero_local(f::F, x, y) where {F} = is_der2_arg1_zero_global(f)
    is_der1_arg2_zero_local(f::F, x, y) where {F} = is_der1_arg2_zero_global(f)
    is_der2_arg2_zero_local(f::F, x, y) where {F} = is_der2_arg2_zero_global(f)
    is_der_cross_zero_local(f::F, x, y) where {F} = is_der_cross_zero_global(f)
    ```

!!! details "Methods you have to implement for 1-to-2 operators"

    | Function                                       | Meaning                                                   |
    |:-----------------------------------------------|:----------------------------------------------------------|
    | `is_der1_out1_zero_local(::typeof(f)) = false` | $\frac{\partial f_1}{\partial x} \neq 0$ for some $x$     | 
    | `is_der2_out1_zero_local(::typeof(f)) = false` | $\frac{\partial^2 f_1}{\partial x^2} \neq 0$ for some $x$ | 
    | `is_der1_out2_zero_local(::typeof(f)) = false` | $\frac{\partial f_2}{\partial x} \neq 0$ for some $x$     | 
    | `is_der2_out2_zero_local(::typeof(f)) = false` | $\frac{\partial^2 f_2}{\partial x^2} \neq 0$ for some $x$ | 

    Optionally, to increase the sparsity of [`TracerLocalSparsityDetector`](@ref), you can additionally implement

    | Function                                          | Meaning                                                    |
    |:--------------------------------------------------|:-----------------------------------------------------------|
    | `is_der1_out1_zero_local(::typeof(f), x) = false` | $\frac{\partial f_1}{\partial x} \neq 0$ for given $x$     | 
    | `is_der2_out1_zero_local(::typeof(f), x) = false` | $\frac{\partial^2 f_1}{\partial x^2} \neq 0$ for given $x$ | 
    | `is_der1_out2_zero_local(::typeof(f), x) = false` | $\frac{\partial f_2}{\partial x} \neq 0$ for given $x$     | 
    | `is_der2_out2_zero_local(::typeof(f), x) = false` | $\frac{\partial^2 f_2}{\partial x^2} \neq 0$ for given $x$ | 

    These fall back to 

    ```julia
    is_der1_out1_zero_local(f::F, x) where {F} = is_der1_out1_zero_global(f)
    is_der2_out1_zero_local(f::F, x) where {F} = is_der2_out1_zero_global(f)
    is_der1_out2_zero_local(f::F, x) where {F} = is_der1_out2_zero_global(f)
    is_der2_out2_zero_local(f::F, x) where {F} = is_der2_out2_zero_global(f)
    ```

### [Generating code](@id code-gen)

After implementing the required classification methods for a function, the function has not been overloaded on our tracer types yet.
SCT provides three functions that generate code via meta-programming:

* 1-to-1: `eval(SCT.generate_code_1_to_1(module_symbol, f))`
* 2-to-1: `eval(SCT.generate_code_1_to_2(module_symbol, f))`
* 1-to-2: `eval(SCT.generate_code_2_to_1(module_symbol, f))`

You are required to call the function that matches your type of operator.

!!! tip "Code generation"
    We will take a look at the code generation mechanism in the example below.

### Example

For some examples on how to overload methods, take a look at our [package extensions](https://github.com/adrhill/SparseConnectivityTracer.jl/tree/main/ext).
Let's look at the `relu` activation function from `ext/SparseConnectivityTracerNNlibExt.jl`, which is a 1-to-1 operator defined as $\text{relu}(x) = \text{max}(0, x)$.

#### Step 1: Classification

The `relu` function has a non-zero first-order derivative $\frac{\partial f}{\partial x}=1$ for inputs $x>0$. 
The second derivative is zero everywhere.
We therefore implement:

```@example overload
import SparseConnectivityTracer as SCT
using NNlib

SCT.is_der1_zero_global(::typeof(relu)) = false
SCT.is_der2_zero_global(::typeof(relu)) = true

SCT.is_der1_zero_local(::typeof(relu), x) = x < 0
```

!!! warning "import SparseConnectivityTracer"
    Note that we imported SCT to extend its operator classification methods on `typeof(relu)`.

#### Step 2: Generating code

The `relu` function has not been overloaded on our tracer types yet.
Let's call the code generation utilities from the [*"Generating code"*](@ref code-gen) section for this purpose:

```@example overload
eval(SCT.generate_code_1_to_1(:NNlib, relu))
```

The `relu` function is now ready to be called with SCT's tracer types.

!!! details "What is the eval call doing?"
    Let's call `generate_code_1_to_1` without wrapping it `eval`:

    ```@example overload
    SCT.generate_code_1_to_1(:NNlib, relu)
    ```

    As you can see, this returns a `quote`, a type of expression containing our generated Julia code.

    **We have to use quotes:** 
    The code generation mechanism lives in SCT, but the generated code has to be evaluated in the package extension, not SCT.
    As you can see in the generated quote, we handle the necessary name-spacing for you.

## Manual overloads

As mentioned above, for functions that take `Real` arguments, manual overloads should generally be avoided.
If such an overload is necessary (e.g. for array inputs), it should follow the following design priciples, ordered by importance:

**Local sparsity detection (`Dual`):**

- Overloads must return conservative sparsity patterns (no false negatives) at the given input.
- Local tracers are allowed to enter branches in user code. User code can be stateful.
- `MethodError`s due to missing overloads can be avoided by returning a very conservative sparsity pattern.

**Global sparsity detection (`GradientTracer` and `HessianTracer`):**

- Overloads must return conservative sparsity patterns (no false negatives) over the entire input domain.
- Tracers must error instead of entering branches in user code. 
  This requires that overloaded functions return tracers instead of `Bool` (or numbers), 
  as the former are designed to error in comparisons.
- Overloads must ignore scalar values of non-tracer inputs.
  *(While not set in stone, changing this rule in the future would require a breaking release.)*
- Sparsity should be prioritized over performance. We assume global sparsity detection can be amortized.
- `MethodError`s due to missing overloads can be avoided by returning a very conservative sparsity pattern.
