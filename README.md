# SparseConnectivityTracer.jl
|               |                                                                     | 
|:--------------|:--------------------------------------------------------------------|
| Documentation | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://adrhill.github.io/SparseConnectivityTracer.jl/dev/) [![Changelog](https://img.shields.io/badge/news-changelog-yellow.svg)](https://github.com/adrhill/SparseConnectivityTracer.jl/blob/main/CHANGELOG.md) |
| Build Status  | [![Build Status](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/adrhill/SparseConnectivityTracer.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/adrhill/SparseConnectivityTracer.jl) [![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a.svg)](https://github.com/aviatesk/JET.jl) |
| Code Style    | [![Code Style: Runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) | 
| Downloads     | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FSparseConnectivityTracer&query=total_requests&suffix=%2Fmonth&label=Downloads)](http://juliapkgstats.com/pkg/SparseConnectivityTracer) [![Dependents](https://juliahub.com/docs/General/SparseConnectivityTracer/stable/deps.svg)](https://juliahub.com/ui/Packages/General/SparseConnectivityTracer?t=2) |
| Citation      | [![arXiv DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2501.17737-red)](https://arxiv.org/abs/2501.17737) [![Zenodo DOI](https://zenodo.org/badge/778978853.svg)](https://zenodo.org/doi/10.5281/zenodo.13138554) |

Fast Jacobian and Hessian sparsity pattern detection via operator-overloading.

> [!TIP]
> If you want to use [automatic sparse differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/) to compute sparse Jacobians or Hessians with SparseConnectivityTracer (instead of just sparsity patterns), 
> refer to the [DifferentiationInterface.jl documentation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/advanced/#Sparsity).

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

By default, `BitSet` is used for internal sparsity pattern representations.
For very large inputs, it might be more efficient to set the type to `Set{UInt}`:

```julia-repl
julia> detector = TracerSparsityDetector(; gradient_pattern_type=Set{UInt})
```

### Hessian

For scalar functions `y = f(x)`, the sparsity pattern of the Hessian can be obtained
by computing a single forward-pass through the function:

```julia-repl
julia> x = rand(4);

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4];

julia> hessian_sparsity(f, x, detector)
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1
```

By default, dictionaries of `BitSet`s are used for internal sparsity pattern representations.
For very large inputs, it might be more efficient to set the type to `Dict{UInt, Set{UInt}}`:

```julia-repl
julia> detector = TracerSparsityDetector(; hessian_pattern_type=Dict{UInt, Set{UInt}})
```

For more detailed examples, take a look at the [documentation](https://adrianhill.de/SparseConnectivityTracer.jl/stable).

### Local tracing

`TracerSparsityDetector` returns conservative "global" sparsity patterns over the entire input domain of `x`. 
It is not compatible with functions that require information about the primal values of a computation (e.g. `iszero`, `>`, `==`).
To compute a less conservative Jacobian or Hessian sparsity pattern at an input point `x`, use `TracerLocalSparsityDetector` instead.
Note that patterns computed with `TracerLocalSparsityDetector` depend on the input `x` and have to be recomputed when `x` changes:

```julia-repl
julia> using SparseConnectivityTracer

julia> detector = TracerLocalSparsityDetector();

julia> f(x) = x[1] > x[2] ? x[1] : x[2];

julia> jacobian_sparsity(f, [1 2], detector)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 ⋅  1

julia> jacobian_sparsity(f, [2 1], detector)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 1  ⋅
```

Take a look at the documentation for [more information on global and local patterns](https://adrianhill.de/SparseConnectivityTracer.jl/stable/user/global_vs_local/).

## ADTypes.jl compatibility
SparseConnectivityTracer uses [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s interface for [sparsity detection](https://sciml.github.io/ADTypes.jl/stable/#Sparsity-detector),
making it compatible with [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl)'s [sparse automatic differentiation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/tutorials/advanced/#Sparsity) functionality.
In fact, the functions `jacobian_sparsity` and `hessian_sparsity` are re-exported from ADTypes.

## Related packages
* [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): automatic sparsity detection via Symbolics.jl and Cassette.jl
* [SparsityTracing.jl](https://github.com/PALEOtoolkit/SparsityTracing.jl): automatic Jacobian sparsity detection using an algorithm based on SparsLinC by Bischof et al. (1996)

## Citation

If you use SparseConnectivityTracer in your research, please cite our TMLR paper [*Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians*](https://openreview.net/forum?id=GtXSN52nIW):

```bibtex
@article{hill2025sparser,
  title={Sparser, Better, Faster, Stronger: Sparsity Detection for Efficient Automatic Differentiation},
  author={Adrian Hill and Guillaume Dalle},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=GtXSN52nIW},
  note={}
}
```

## Acknowledgements

Adrian Hill gratefully acknowledges funding from the German Federal Ministry of Education and Research under the grant BIFOLD25B.
