module SparseConnectivityTracerRecursiveArrayToolsExt

import SparseConnectivityTracer as SCT
using RecursiveArrayTools: ArrayPartition, NamedArrayPartition

function SCT.trace_input(
    ::Type{T}, xs::ArrayPartition, i
) where {T<:Union{SCT.AbstractTracer,SCT.Dual}}
    ts = SCT.create_tracers(T, xs, eachindex(xs))
    lengths = map(length, xs.x)
    length_sums = (0, cumsum(lengths)...)
    return ArrayPartition(
        Tuple(
            reshape(view(ts, (1 + length_sums[j]):(length_sums[j + 1])), size(xs.x[j])) for
            j in eachindex(xs.x)
        ),
    )
end

function SCT.trace_input(
    ::Type{T}, xs::NamedArrayPartition, i
) where {T<:Union{SCT.AbstractTracer,SCT.Dual}}
    return NamedArrayPartition(
        SCT.trace_input(T, getfield(xs, :array_partition), i),
        getfield(xs, :names_to_indices),
    )
end

end # module SparseConnectivityTracerRecursiveArrayToolsExt