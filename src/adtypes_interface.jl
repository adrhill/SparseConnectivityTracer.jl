#= This file implements the ADTypes interface for `AbstractSparsityDetector`s =#

const DEFAULT_GRADIENT_TRACER = GradientTracer{IndexSetGradientPattern{Int,BitSet}}
const DEFAULT_HESSIAN_TRACER = HessianTracer{
    DictHessianPattern{Int,BitSet,Dict{Int,BitSet},NotShared}
}

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

Local sparsity patterns are less convervative than global patterns and need to be recomputed for each input `x`:

```jldoctest
julia> using SparseConnectivityTracer

julia> method = TracerLocalSparsityDetector();

julia> f(x) = x[1] * x[2]; # J_f = [x[2], x[1]]

julia> jacobian_sparsity(f, [1, 0], method)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 ⋅  1

julia> jacobian_sparsity(f, [0, 1], method)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 1  ⋅

julia> jacobian_sparsity(f, [0, 0], method)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 0 stored entries:
 ⋅  ⋅

julia> jacobian_sparsity(f, [1, 1], method)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 2 stored entries:
 1  1
```

`TracerLocalSparsityDetector` can compute sparsity patterns of functions that contain comparisons and `ifelse` statements:


```jldoctest
julia> f(x) = x[1] > x[2] ? x[1:3] : x[2:4];

julia> jacobian_sparsity(f, [1, 2, 3, 4], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1

julia> jacobian_sparsity(f, [2, 1, 3, 4], TracerLocalSparsityDetector())
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 3 stored entries:
 1  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅
```

```jldoctest
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
