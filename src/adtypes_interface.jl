#= This file implements the ADTypes interface for `AbstractSparsityDetector`s =#

const DEFAULT_GRADIENT_PATTERN = IndexSetGradientPattern{Int,BitSet}
const DEFAULT_GRADIENT_TRACER = GradientTracer{DEFAULT_GRADIENT_PATTERN}

const DEFAULT_HESSIAN_PATTERN = DictHessianPattern{Int,BitSet,Dict{Int,BitSet},NotShared}
const DEFAULT_HESSIAN_TRACER = HessianTracer{DEFAULT_HESSIAN_PATTERN}

"""
    TracerSparsityDetector <: ADTypes.AbstractSparsityDetector

Singleton struct for integration with the sparsity detection framework of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

Computes global sparsity patterns over the entire input domain.
For local sparsity patterns at a specific input point, use [`TracerLocalSparsityDetector`](@ref).

# Example

```jldoctest
julia> using SparseConnectivityTracer

julia> detector = TracerSparsityDetector()
TracerSparsityDetector()

julia> jacobian_sparsity(diff, rand(4), detector)
3×4 SparseArrays.SparseMatrixCSC{Bool, Int64} with 6 stored entries:
 1  1  ⋅  ⋅
 ⋅  1  1  ⋅
 ⋅  ⋅  1  1

julia> f(x) = x[1] + x[2]*x[3] + 1/x[4];

julia> hessian_sparsity(f, rand(4), detector)
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

julia> detector = TracerLocalSparsityDetector()
TracerLocalSparsityDetector()

julia> f(x) = x[1] * x[2]; # J_f = [x[2], x[1]]

julia> jacobian_sparsity(f, [1, 0], detector)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 ⋅  1

julia> jacobian_sparsity(f, [0, 1], detector)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 1 stored entry:
 1  ⋅

julia> jacobian_sparsity(f, [0, 0], detector)
1×2 SparseArrays.SparseMatrixCSC{Bool, Int64} with 0 stored entries:
 ⋅  ⋅

julia> jacobian_sparsity(f, [1, 1], detector)
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

## Pretty printing
for detector in (:TracerSparsityDetector, :TracerLocalSparsityDetector)
    @eval function Base.show(io::IO, d::$detector{TG,TH}) where {TG,TH}
        if TG == DEFAULT_GRADIENT_TRACER && TH == DEFAULT_HESSIAN_TRACER
            print(io, $detector, "()")
        else
            print(io, $detector, "{", TG, ",", TH, "}()")
        end
        return nothing
    end
end

## Stable API to allow packages like DI to allocate caches of tracers

"""
    jacobian_tracer_type(x, detector)

Return the tracer type used for Jacobian sparsity detection.
"""
jacobian_tracer_type(x, ::TracerSparsityDetector{TG}) where {TG} = TG
jacobian_tracer_type(x, ::TracerLocalSparsityDetector{TG}) where {TG} = Dual{eltype(x),TG}

"""
    hessian_tracer_type(x, detector)

Return the tracer type used for Hessian sparsity detection.
"""
hessian_tracer_type(x, ::TracerSparsityDetector{TG,TH}) where {TG,TH} = TH
function hessian_tracer_type(x, ::TracerLocalSparsityDetector{TG,TH}) where {TG,TH}
    return Dual{eltype(x),TH}
end

"""
    jacobian_buffer(x, detector)

Allocate a buffer similiar to `x` with the required tracer type for Jacobian sparsity detection.
Thin wrapper around `similar` that doesn't expose internal types.
"""
function jacobian_buffer(
    x, detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector}
)
    return similar(x, jacobian_tracer_type(x, detector))
end

"""
    hessian_buffer(x, detector)

Allocate a buffer similiar to `x` with the required tracer type for Hessian sparsity detection.
Thin wrapper around `similar` that doesn't expose internal types.
"""
function hessian_buffer(
    x, detector::Union{TracerSparsityDetector,TracerLocalSparsityDetector}
)
    return similar(x, hessian_tracer_type(x, detector))
end
