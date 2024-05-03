# SparseConnectivityTracer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/dev/)
[![Build Status](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Fast Jacobian and Hessian sparsity detection via operator-overloading.

> [!WARNING]
> This package is in early development. Expect frequent breaking changes and refer to the stable documentation.

## Installation 
To install this package, open the Julia REPL and run 

```julia-repl
julia> ]add SparseConnectivityTracer
```

## Examples
### Jacobian

For functions `y = f(x)` and `f!(y, x)`, the sparsity pattern of the Jacobian of $f$ can be obtained
by computing a single forward-pass through `f`:

```julia-repl
julia> using SparseConnectivityTracer

julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> jacobian_pattern(f, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```

As a larger example, let's compute the sparsity pattern from a convolutional layer from [Flux.jl](https://github.com/FluxML/Flux.jl):
```julia-repl
julia> using SparseConnectivityTracer, Flux

julia> x = rand(28, 28, 3, 1);

julia> layer = Conv((3, 3), 3 => 2);

julia> jacobian_pattern(layer, x)
1352×2352 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 36504 stored entries:
⎡⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎤
⎢⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠙⢿⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⣀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⣷⣄⠀⎥
⎢⢤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠛⢦⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠳⣤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠓⎥
⎢⠀⠙⢿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠉⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⣄⠀⠀⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⣷⣄⠀⠀⠀⠀⎥
⎢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⠀⠀⎥
⎣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⢿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣄⎦
```

The type of index set `T` that is internally used to keep track of connectivity can be specified via `jacobian_pattern(f, x, T)`, defaulting to `BitSet`. 
For high-dimensional functions, `Set{UInt64}` can be more efficient .

### Hessian

For scalar functions `y = f(x)`, the sparsity pattern of the Hessian of $f$ can be obtained
by computing a single forward-pass through `f`:

```julia-repl
julia> x = rand(5);

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4] + 1*x[5];

julia> hessian_pattern(f, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> g(x) = f(x) + x[2]^x[5];

julia> hessian_pattern(g, x)
5×5 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 7 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
```

For more detailled examples, take a look at the [documentation](https://adrianhill.de/SparseConnectivityTracer.jl/dev).

## Related packages
* [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): automatic sparsity detection via Symbolics.jl and Cassette.jl
* [SparsityTracing.jl](https://github.com/PALEOtoolkit/SparsityTracing.jl): automatic Jacobian sparsity detection using an algorithm based on SparsLinC by Bischof et al. (1996)
