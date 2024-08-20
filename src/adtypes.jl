"""
    TracerSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

Computes global sparsity patterns over the entire input domain.
For local sparsity patterns at a specific input point, use [`TracerLocalSparsityDetector`](@ref).

# Example

```jldoctest
julia> using SparseConnectivityTracer

julia> jacobian_sparsity(diff, rand(4), TracerSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 6 stored entries:
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  1  1
```

```jldoctest
julia> using SparseConnectivityTracer

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4];

julia> hessian_sparsity(f, rand(4), TracerSparsityDetector())
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1
```
"""
struct TracerSparsityDetector{TG<:GradientTracer,TH<:HessianTracer} <:
       ADTypes.AbstractSparsityDetector end
function TracerSparsityDetector(
    ::Type{TG}, ::Type{TH}
) where {TG<:GradientTracer,TH<:HessianTracer}
    return TracerSparsityDetector{TG,TH}()
end
function TracerSparsityDetector(;
    gradient_tracer_type::Type{TG}=DEFAULT_GRADIENT_TRACER,
    hessian_tracer_type::Type{TH}=DEFAULT_HESSIAN_TRACER,
) where {TG<:GradientTracer,TH<:HessianTracer}
    return TracerSparsityDetector(gradient_tracer_type, hessian_tracer_type)
end

function ADTypes.jacobian_sparsity(f, x, ::TracerSparsityDetector{TG,TH}) where {TG,TH}
    return _jacobian_sparsity(f, x, TG)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::TracerSparsityDetector{TG,TH}) where {TG,TH}
    return _jacobian_sparsity(f!, y, x, TG)
end

function ADTypes.hessian_sparsity(f, x, ::TracerSparsityDetector{TG,TH}) where {TG,TH}
    return _hessian_sparsity(f, x, TH)
end

"""
    TracerLocalSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

Computes local sparsity patterns at an input point `x`.
For global sparsity patterns, use [`TracerSparsityDetector`](@ref).

# Example

```jldoctest
julia> using SparseConnectivityTracer

julia> f(x) = x[1] > x[2] ? x[1:3] : x[2:4];

julia> jacobian_sparsity(f, [1.0, 2.0, 3.0, 4.0], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1

julia> jacobian_sparsity(f, [2.0, 1.0, 3.0, 4.0], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
```

```jldoctest
julia> using SparseConnectivityTracer

julia> f(x) = x[1] + max(x[2], x[3]) * x[3] + 1/x[4];

julia> hessian_sparsity(f, [1.0, 2.0, 3.0, 4.0], TracerLocalSparsityDetector())
4×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1
```
"""
struct TracerLocalSparsityDetector{TG<:GradientTracer,TH<:HessianTracer} <:
       ADTypes.AbstractSparsityDetector end
function TracerLocalSparsityDetector(
    ::Type{TG}, ::Type{TH}
) where {TG<:GradientTracer,TH<:HessianTracer}
    return TracerLocalSparsityDetector{TG,TH}()
end
function TracerLocalSparsityDetector(;
    gradient_tracer_type::Type{TG}=DEFAULT_GRADIENT_TRACER,
    hessian_tracer_type::Type{TH}=DEFAULT_HESSIAN_TRACER,
) where {TG<:GradientTracer,TH<:HessianTracer}
    return TracerLocalSparsityDetector(gradient_tracer_type, hessian_tracer_type)
end

function ADTypes.jacobian_sparsity(f, x, ::TracerLocalSparsityDetector{TG,TH}) where {TG,TH}
    return _local_jacobian_sparsity(f, x, TG)
end

function ADTypes.jacobian_sparsity(
    f!, y, x, ::TracerLocalSparsityDetector{TG,TH}
) where {TG,TH}
    return _local_jacobian_sparsity(f!, y, x, TG)
end

function ADTypes.hessian_sparsity(f, x, ::TracerLocalSparsityDetector{TG,TH}) where {TG,TH}
    return _local_hessian_sparsity(f, x, TH)
end
