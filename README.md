# SparseConnectivityTracer.jl
|               |                                                                     | 
|:--------------|:--------------------------------------------------------------------|
| Documentation | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/dev/) [![Changelog](https://img.shields.io/badge/news-changelog-yellow.svg)](https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/CHANGELOG.md) |
| Build Status  | [![Build Status](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a.svg)](https://github.com/aviatesk/JET.jl) |
| Code Style    | [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) | 
| Downloads     | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FSparseConnectivityTracer&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/SparseConnectivityTracer) [![Dependents](https://juliahub.com/docs/General/SparseConnectivityTracer/stable/deps.svg)](https://juliahub.com/ui/Packages/General/SparseConnectivityTracer?t=2) |
| Citation      | [![DOI](https://zenodo.org/badge/778978853.svg)](https://zenodo.org/doi/10.5281/zenodo.13138554) |

Fast Jacobian and Hessian sparsity detection via operator-overloading.

## Installation 
To install this package, open the Julia REPL and run 

```julia-repl
julia> ]add SparseConnectivityTracer
```

## Examples
### Jacobian

For functions `y = f(x)` and `f!(y, x)`, the sparsity pattern of the Jacobian can be obtained
by computing a single forward-pass through the function:

```julia-repl
julia> using SparseConnectivityTracer

julia> detector = TracerSparsityDetector();

julia> x = rand(3);

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> jacobian_sparsity(f, x, detector)
3×3 SparseArrays.SparseMatrixCSC{Bool, Int64} with 4 stored entries:
 1  ⋅  ⋅
 1  1  ⋅
 ⋅  ⋅  1
```

As a larger example, let's compute the sparsity pattern from a convolutional layer from [Flux.jl](https://github.com/FluxML/Flux.jl):

```julia-repl
julia> using SparseConnectivityTracer, Flux

julia> detector = TracerSparsityDetector();

julia> x = rand(28, 28, 3, 1);

julia> layer = Conv((3, 3), 3 => 2);

julia> jacobian_sparsity(layer, x, detector)
1352×2352 SparseArrays.SparseMatrixCSC{Bool, Int64} with 36504 stored entries:
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

### Hessian

For scalar functions `y = f(x)`, the sparsity pattern of the Hessian of $f$ can be obtained
by computing a single forward-pass through `f`:

```julia-repl
julia> x = rand(5);

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4] + 1*x[5];

julia> hessian_sparsity(f, x, detector)
5×5 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅

julia> g(x) = f(x) + x[2]^x[5];

julia> hessian_sparsity(g, x, detector)
5×5 SparseArrays.SparseMatrixCSC{Bool, Int64} with 7 stored entries:
 ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅  1
```

For more detailed examples, take a look at the [documentation](https://adrianhill.de/SparseConnectivityTracer.jl/dev).

### Local tracing

`TracerSparsityDetector` returns conservative sparsity patterns over the entire input domain of `x`. 
It is not compatible with functions that require information about the primal values of a computation (e.g. `iszero`, `>`, `==`).

To compute a less conservative sparsity pattern at an input point `x`, use `TracerLocalSparsityDetector` instead.
Note that patterns computed with `TracerLocalSparsityDetector` depend on the input `x` and have to be recomputed when `x` changes:

```julia-repl
julia> using SparseConnectivityTracer

julia> detector = TracerLocalSparsityDetector();

julia> f(x) = ifelse(x[2] < x[3], x[1] ^ x[2], x[3] * x[4]);

julia> hessian_sparsity(f, [1 2 3 4], detector)
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 4 stored entries:
 1  1  ⋅  ⋅
 1  1  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅

julia> hessian_sparsity(f, [1 3 2 4], detector)
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  1
 ⋅  ⋅  1  ⋅
```

## ADTypes.jl compatibility
SparseConnectivityTracer uses [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s interface for [sparsity detection](https://sciml.github.io/ADTypes.jl/stable/#Sparsity-detector),
making it compatible with [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)'s [sparse automatic differentiation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/advanced/#Sparsity) functionality.
In fact, the functions `jacobian_sparsity` and `hessian_sparsity` are re-exported from ADTypes.

## Related packages
* [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): automatic sparsity detection via Symbolics.jl and Cassette.jl
* [SparsityTracing.jl](https://github.com/PALEOtoolkit/SparsityTracing.jl): automatic Jacobian sparsity detection using an algorithm based on SparsLinC by Bischof et al. (1996)

## Acknowledgements

Adrian Hill acknowledges support by the Federal Ministry of Education and Research (BMBF) for the Berlin Institute for the Foundations of Learning and Data (BIFOLD) (01IS18037A).
