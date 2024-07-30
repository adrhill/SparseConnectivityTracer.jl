# SparseConnectivityTracer.jl

## Version `v0.6.0`
* ![BREAKING][badge-breaking] Remove `ConnectivityTracer` ([#140])
* ![BREAKING][badge-breaking] Remove legacy interface ([#140])
    * instead of `jacobian_pattern(f, x)`, use `jacobian_sparsity(f, x, TracerSparsityDetector())`
    * instead of `hessian_pattern(f, x)`, use `hessian_sparsity(f, x, TracerSparsityDetector())`
    * instead of `local_jacobian_pattern(f, x)`, use `jacobian_sparsity(f, x, TracerLocalSparsityDetector())`
    * instead of `local_hessian_pattern(f, x)`, use `hessian_sparsity(f, x, TracerLocalSparsityDetector())`
* ![Bugfix][badge-bugfix] Remove overloads on `similar` to reduce amount of invalidations ([#132])
* ![Bugfix][badge-bugfix] Fix sparse array construction ([#142])
* ![Enhancement][badge-enhancement] Add array overloads ([#131])
* ![Enhancement][badge-enhancement] Generalize sparsity pattern representations ([#139], [#119])
* ![Enhancement][badge-enhancement] Reduce allocations of new tracers ([#128])
* ![Enhancement][badge-enhancement] Reduce compile times ([#119])

[#142]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/142
[#140]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/140
[#139]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/139
[#132]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/132
[#131]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/131
[#128]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/128
[#126]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/126
[#119]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/119

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg