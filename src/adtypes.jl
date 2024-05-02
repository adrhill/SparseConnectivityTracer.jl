"""
    TracerSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of ADTypes.jl.

# Example

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> ADTypes.jacobian_sparsity(diff, rand(4), TracerSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 6 stored entries:
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  1  1
```

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4];

julia> ADTypes.hessian_sparsity(f, rand(4), TracerSparsityDetector())
4×4 SparseArrays.SparseMatrixCSC{Bool, UInt64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1
```
"""
struct TracerSparsityDetector{S<:AbstractIndexSet} <: ADTypes.AbstractSparsityDetector end
TracerSparsityDetector(::Type{S}) where {S<:AbstractIndexSet} = TracerSparsityDetector{S}()
TracerSparsityDetector() = TracerSparsityDetector(BitSet)

function ADTypes.jacobian_sparsity(
    f, x, ::TracerSparsityDetector{S}
) where {S<:AbstractIndexSet}
    return pattern(f, JacobianTracer{S}, x)
end

function ADTypes.jacobian_sparsity(
    f!, y, x, ::TracerSparsityDetector{S}
) where {S<:AbstractIndexSet}
    return pattern(f!, y, JacobianTracer{S}, x)
end

function ADTypes.hessian_sparsity(
    f, x, ::TracerSparsityDetector{S}
) where {S<:AbstractIndexSet}
    return pattern(f, HessianTracer{S}, x)
end
