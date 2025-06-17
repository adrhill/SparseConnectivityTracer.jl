# SparseConnectivityTracer.jl

## Version `v0.6.21`
* ![Documentation][badge-docs] Document limitations on stateful code ([#249])

## Version `v0.6.20`
* ![Bugfix][badge-bugfix] Revert PR [#243] due to concerns of non-spare patterns ([#248])

## Version `v0.6.19`
* ![Enhancement][badge-enhancement] Stop tracing through multiplication by zero ([#243])
* ![Maintenance][badge-maintenance] Update code style and formatter to Runic.jl

## Version `v0.6.18`
* ![Maintenance][badge-maintenance] DataInterpolations `v8` compatiblity ([#234])

## Version `v0.6.17`
* ![Enhancement][badge-enhancement] Performance optimization in pattern creation ([#239])

## Version `v0.6.16`
* ![Feature][badge-feature] Add more matrix division methods ([#236])

## Version `v0.6.15`
* ![Feature][badge-feature] Add stable API for tracer type via `jacobian_eltype` and `hessian_eltype` ([#233])

## Version `v0.6.14`
* ![Feature][badge-feature] Add stable API for allocation of buffers via `jacobian_buffer` and `hessian_buffer` ([#232])

## Version `v0.6.13`
* ![Bugfix][badge-bugfix] Return `Dual` on `zero` and friends ([#231])

## Version `v0.6.12`
* ![Bugfix][badge-bugfix] Fix for method ambiguities resulting from the 3-argument `dot` methods introduced in `v0.6.11` ([#228])

## Version `v0.6.11`
* ![Documentation][badge-docs] SCT has a new preprint! 🎉 
  Check it out on the arXiv: [*Sparser, Better, Faster, Stronger: Efficient Automatic Differentiation for Sparse Jacobians and Hessians*](https://arxiv.org/abs/2501.17737) ([#225])
* ![Feature][badge-feature] Add overloads for 3-argument `dot` ([#226])

## Version `v0.6.10`
* ![Bugfix][badge-bugfix] Fix `jacobian_sparsity` output initialization for  inplace functions ([#223])

## Version `v0.6.9`
* ![Bugfix][badge-bugfix] Relax type annotations in Jacobian output parsing ([#217])
* ![Enhancement][badge-enhancement] Simplify DataInterpolations.jl extension ([#210])

## Version `v0.6.8`

* ![Feature][badge-feature] Support `clamp` and `clamp!` ([#208])
* ![Maintenance][badge-maintenance] Remove internal set type `DuplicateVector` ([#209])

## Version `v0.6.7`

* ![Enhancement][badge-enhancement] Drop compatibility with Julia <1.10 to improve tracer performance ([#204], [#205])

## Version `v0.6.6`

* ![Bugfix][badge-bugfix] Fix detector display by replacing `println` with `print` ([#201])
* ![Enhancement][badge-enhancement] Improve code generation for 2-to-1 overloads on arbitrary types ([#197], [#202])
* ![Maintenance][badge-maintenance] Update package tests and CI workflow ([#198], [#199])

## Version `v0.6.5`

* ![Bugfix][badge-bugfix] Fix LogExpFunctions.jl compat entry ([#195])
* ![Documentation][badge-docs] Fix "How it works" documentation ([#193])

## Version `v0.6.4`

* ![Enhancement][badge-enhancement] Shorter printing of default detectors ([#190])
* ![Documentation][badge-docs] Consistently refer to `TracerSparsityDetector` as `detector` ([#191])
* ![Maintenance][badge-maintenance] Make imports explicit, test with ExplicitImports.jl ([#188])

## Version `v0.6.3`

* ![Feature][badge-feature] Add DataInterpolations.jl package extension ([#178])
* ![Feature][badge-feature] Add LogExpFunctions.jl package extension ([#184])
* ![Feature][badge-feature] Add NaNMath.jl package extension ([#187])
* ![Feature][badge-feature] Support two-argument `atan` and `log` ([#185])
* ![Documentation][badge-docs] Document limitations of operator overloading utils ([#180])
* ![Documentation][badge-docs] Reexport ADTypes interface ([#182])
* ![Documentation][badge-docs] Update developer documentation URLs ([#186])
* ![Maintenance][badge-maintenance] Reorganize code and update code generation utilities ([#179], [#183])

## Version `v0.6.2`

* ![Feature][badge-feature] Return only primal value when applying non-differentiable methods to `Dual` numbers ([#169])
* ![Feature][badge-feature] Increase sparsity for Diagonal inputs ([#165])
* ![Feature][badge-feature] Add more methods on `round`, `rand` ([#162])
* ![Documentation][badge-docs] This release brings large updates to the documentation:
  * Document limitations ([#175])
  * Document global vs. local patterns ([#176])
  * Add "How it works" developer documentation ([#174])
  * Add developer documentation on custom overloads ([#177])
* ![Enhancement][badge-enhancement] Refactor type conversions ([#173], [#168], [#166])
* ![Enhancement][badge-enhancement] Make comparisons regular operators ([#169])
* ![Bugfix][badge-bugfix] Fix Hessian on NNlib activation functions `celu`, `elu`, `selu`, `hardswish` ([#162])
* ![Bugfix][badge-bugfix] Fix `isless` ([#161])

## Version `v0.6.1`

* ![Enhancement][badge-enhancement] Improve the performance of Hessian pattern tracing by an order of magnitude:
  * Internally represent Hessian patterns with dictionaries ([#155], [#158])
  * Increase performance via symmetry of Hessian ([#151])
* ![Feature][badge-feature] Support ComponentArrays ([#146])
* ![Feature][badge-feature] Support boolean not (`!`) ([#150])
* ![Feature][badge-feature] Support `isless` ([#149])

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

[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg

[#249]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/249
[#248]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/248
[#243]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/243
[#239]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/239
[#236]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/236
[#234]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/234
[#233]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/233
[#232]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/232
[#231]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/231
[#228]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/228
[#226]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/226
[#225]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/225
[#223]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/223
[#217]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/217
[#210]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/210
[#209]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/209
[#208]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/208
[#205]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/205
[#204]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/204
[#202]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/202
[#201]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/201
[#199]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/199
[#198]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/198
[#197]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/197
[#195]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/195
[#193]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/193
[#191]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/191
[#190]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/190
[#188]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/188
[#186]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/186
[#185]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/185
[#184]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/184
[#183]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/183
[#182]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/182
[#180]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/180
[#179]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/179
[#178]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/178
[#177]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/177
[#176]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/176
[#175]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/175
[#174]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/174
[#173]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/173
[#169]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/169
[#168]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/168
[#166]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/166
[#165]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/165
[#162]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/162
[#161]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/161
[#158]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/158
[#155]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/155
[#151]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/151
[#150]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/150
[#149]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/149
[#146]: https://github.com/adrhill/SparseConnectivityTracer.jl/pull/146
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
