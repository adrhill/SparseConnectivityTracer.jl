using SnoopCompileCore
invalidations = @snoopr using ADTypes, SparseConnectivityTracer
tinf = @snoopi_deep begin
    include("test/britgas_definition.jl")
    ADTypes.hessian_sparsity(sum_britgas_cons, zeros(450), TracerSparsityDetector())
end;
using SnoopCompile
trees = invalidation_trees(invalidations)
staletrees = precompile_blockers(trees, tinf)

@show length(uinvalidated(invalidations))  # show total invalidations

show(trees[end])  # show the most invalidating method

# Count number of children (number of invalidations per invalidated method)
n_invalidations = map(SnoopCompile.countchildren, trees)

# (optional) plot the number of children per method invalidations
import Plots
Plots.plot(
    1:length(trees),
    n_invalidations;
    markershape=:circle,
    xlabel="i-th method invalidation",
    label="Number of children per method invalidations"
)

# (optional) report invalidations summary
using PrettyTables  # needed for `report_invalidations` to be defined
SnoopCompile.report_invalidations(;
     invalidations,
     process_filename = x -> last(split(x, ".julia/packages/")),
     n_rows = 0,  # no-limit (show all invalidations)
  )
