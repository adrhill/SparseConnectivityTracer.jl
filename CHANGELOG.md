# SparseConnectivityTracer.jl

## Version `v0.6.0`
* ![BREAKING][badge-breaking] Remove `ConnectivityTracer` ([#140][pr-140])
* ![Bugfix][badge-bugfix] Remove overloads on `similar` to reduce amount of invalidations  ([#132][pr-132])
* ![Enhancement][badge-enhancement] Add array overloads ([#131][pr-131])
* ![Enhancement][badge-enhancement] Generalize sparsity pattern representations ([#139][pr-139], [#119][pr-119])
* ![Enhancement][badge-enhancement] Reduce allocations of new tracers ([#128][pr-128])
* ![Enhancement][badge-enhancement] Reduce compile times ([#119][pr-119])

[pr-140]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/140
[pr-139]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/139
[pr-132]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/132
[pr-131]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/131
[pr-128]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/128
[pr-126]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/126
[pr-119]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/119

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