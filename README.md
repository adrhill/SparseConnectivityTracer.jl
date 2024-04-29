# SparseConnectivityTracer.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/dev/)
[![Build Status](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Fast Jacobian and Hessian sparsity detection via operator-overloading.

## Installation 
To install this package, open the Julia REPL and run 
```julia-repl
julia> ]add SparseConnectivityTracer
```

## Examples

```julia-repl
julia> using SparseConnectivityTracer

julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> pattern(f, JacobianTracer, x)
3×3 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```

As a larger example, let's compute the sparsity pattern from a convolutional layer from [Flux.jl](https://github.com/FluxML/Flux.jl):
```julia-repl
julia> using SparseConnectivityTracer, Flux

julia> x = rand(28, 28, 3, 1);

julia> layer = Conv((3, 3), 3 => 8);

julia> pattern(layer, JacobianTracer, x)
5408×2352 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 146016 stored entries:
⎡⠙⢦⡀⠀⠀⠘⢷⣄⠀⠀⠈⠻⣦⡀⠀⠀⠀⎤
⎢⠀⠀⠙⢷⣄⠀⠀⠙⠷⣄⠀⠀⠈⠻⣦⡀⠀⎥
⎢⢶⣄⠀⠀⠙⠳⣦⡀⠀⠈⠳⢦⡀⠀⠈⠛⠂⎥
⎢⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⠀⠙⢦⣄⠀⠀⎥
⎢⣀⡀⠀⠉⠳⣄⡀⠀⠈⠻⣦⣀⠀⠀⠙⢷⡄⎥
⎢⠈⠻⣦⡀⠀⠈⠛⢦⡀⠀⠀⠙⢷⣄⠀⠀⠀⎥
⎢⠀⠀⠈⠻⣦⡀⠀⠀⠙⢷⣄⠀⠀⠙⠷⣄⠀⎥
⎢⠻⣦⡀⠀⠈⠙⢷⣄⠀⠀⠉⠻⣦⡀⠀⠈⠁⎥
⎢⠀⠀⠙⢦⣀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⎥
⎢⢤⣄⠀⠀⠙⠳⣄⡀⠀⠉⠳⣤⡀⠀⠈⠛⠂⎥
⎢⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⠈⠙⢦⡀⠀⠀⎥
⎢⣀⠀⠀⠙⢷⣄⡀⠀⠈⠻⣦⣀⠀⠀⠙⢷⡄⎥
⎢⠈⠳⣦⡀⠀⠈⠻⣦⡀⠀⠀⠙⢷⣄⠀⠀⠀⎥
⎢⠀⠀⠈⠻⣦⡀⠀⠀⠙⢦⣄⠀⠀⠙⢷⣄⠀⎥
⎢⠻⣦⡀⠀⠈⠙⢷⣄⠀⠀⠉⠳⣄⡀⠀⠉⠁⎥
⎢⠀⠈⠛⢦⡀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⡀⠀⎥
⎢⢤⣄⠀⠀⠙⠶⣄⠀⠀⠙⠷⣤⡀⠀⠈⠻⠆⎥
⎢⠀⠙⢷⣄⠀⠀⠈⠳⣦⡀⠀⠈⠻⣦⡀⠀⠀⎥
⎣⠀⠀⠀⠙⢷⣄⠀⠀⠈⠻⣦⠀⠀⠀⠙⢦⡀⎦
```

SparseConnectivityTracer enumerates inputs `x` and primal outputs `y = f(x)` and returns a sparse matrix `C` of size $m \times n$, where `C[i, j]` is `true` if the compute graph connects the $j$-th entry in `x` to the $i$-th entry in `y`.

For more detailled examples, take a look at the [documentation](https://adrianhill.de/SparseConnectivityTracer.jl/dev).

## Related packages
* [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): automatic sparsity detection via Symbolics.jl and Cassette.jl
* [SparsityTracing.jl](https://github.com/PALEOtoolkit/SparsityTracing.jl): automatic Jacobian sparsity detection using an algorithm based on SparsLinC by Bischof et al. (1996)