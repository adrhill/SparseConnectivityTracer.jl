"""
    TracerSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

Computes global sparsity patterns over the entire input domain.
For local sparsity patterns at a specific input point, use [`TracerLocalSparsityDetector`](@ref).

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
function TracerSparsityDetector(::Type{G}) where {G}
    return TracerSparsityDetector{G,Set{Tuple{Int,Int}}}()
end
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

"""
    TracerLocalSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

Computes local sparsity patterns at an input point `x`.
For global sparsity patterns, use [`TracerSparsityDetector`](@ref).

# Example

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> f(x) = x[1] > x[2] ? x[1:3] : x[2:4];

julia> ADTypes.jacobian_sparsity(f, [1.0, 2.0, 3.0, 4.0], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1

julia> ADTypes.jacobian_sparsity(f, [2.0, 1.0, 3.0, 4.0], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
```

```jldoctest
julia> using ADTypes, SparseConnectivityTracer

julia> f(x) = x[1] + max(x[2], x[3]) * x[3] + 1/x[4];

julia> ADTypes.hessian_sparsity(f, [1.0, 2.0, 3.0, 4.0], TracerLocalSparsityDetector())
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1
```
"""
struct TracerLocalSparsityDetector{G,H} <: ADTypes.AbstractSparsityDetector end
function TracerLocalSparsityDetector(::Type{G}, ::Type{H}) where {G,H}
    return TracerLocalSparsityDetector{G,H}()
end
function TracerLocalSparsityDetector(::Type{G}) where {G}
    return TracerLocalSparsityDetector{G,Set{Tuple{Int,Int}}}()
end
TracerLocalSparsityDetector() = TracerLocalSparsityDetector(BitSet, Set{Tuple{Int,Int}})

function ADTypes.jacobian_sparsity(f, x, ::TracerLocalSparsityDetector{G,H}) where {G,H}
    return local_jacobian_pattern(f, x, G)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::TracerLocalSparsityDetector{G,H}) where {G,H}
    return local_jacobian_pattern(f!, y, x, G)
end

function ADTypes.hessian_sparsity(f, x, ::TracerLocalSparsityDetector{G,H}) where {G,H}
    return local_hessian_pattern(f, x, G, H)
end
