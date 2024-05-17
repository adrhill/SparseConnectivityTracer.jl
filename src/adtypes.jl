"""
    TracerSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

# Example

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> ADTypes.jacobian_sparsity(diff, rand(4), TracerSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 6 stored entries:
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  1  1
```

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4];

julia> ADTypes.hessian_sparsity(f, rand(4), TracerSparsityDetector())
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1
```
"""
struct TracerSparsityDetector{G,H} <: ADTypes.AbstractSparsityDetector end
TracerSparsityDetector(::Type{G}, ::Type{H}) where {G,H} = TracerSparsityDetector{G,H}()
TracerSparsityDetector(::Type{G}) where {G} = TracerSparsityDetector{G,Dict{eltype(G),G}}()
TracerSparsityDetector() = TracerSparsityDetector(BitSet, Set{Tuple{Int,Int}})

function ADTypes.jacobian_sparsity(f, x, ::TracerSparsityDetector{G,H}) where {G,H}
    return jacobian_pattern(f, x, G)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::TracerSparsityDetector{G,H}) where {G,H}
    return jacobian_pattern(f!, y, x, G)
end

function ADTypes.hessian_sparsity(f, x, ::TracerSparsityDetector{G,H}) where {G,H}
    return hessian_pattern(f, x, G, H)
end
