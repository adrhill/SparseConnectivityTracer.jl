using SnoopCompileCore: @snoop_invalidations
invalidations = @snoop_invalidations begin
    import SparseConnectivityTracer
end

using SnoopCompile: SnoopCompile, filtermod, invalidation_trees, uinvalidated
inv_owned = length(filtermod(SparseConnectivityTracer, invalidation_trees(invalidations)))
inv_total = length(uinvalidated(invalidations))
inv_deps = inv_total - inv_owned

@show inv_total, inv_deps

# Report invalidations summary:
import PrettyTables  # needed for `report_invalidations` to be defined
SnoopCompile.report_invalidations(;
    invalidations,
    process_filename = x -> last(split(x, ".julia/packages/")),
    n_rows = 0,
)
