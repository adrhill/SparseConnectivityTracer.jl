#= This file implements the ADTypes interface for `AbstractSparsityDetector`s =#
const DEFAULT_SET_TYPE = BitSet
const DEFAULT_DICT_TYPE = Dict{eltype(DEFAULT_SET_TYPE), DEFAULT_SET_TYPE}
const DEFAULT_SHARED_TYPE = NotShared

# TODO: document
function gradient_tracer_type(
        ::Type{G} = DEFAULT_SET_TYPE
    ) where {G <: AbstractSet}
    I = eltype(G)
    return GradientTracer{I, G}
end

# TODO: document
function hessian_tracer_type(
        ::Type{H} = DEFAULT_DICT_TYPE, ::Type{S} = DEFAULT_SHARED_TYPE
    ) where {H <: AbstractDict, S <: SharingBehavior}
    I = keytype(H)
    G = valtype(H)
    return HessianTracer{I, G, H, S}
end

function hessian_tracer_type(
        ::Type{H}, ::Type{S} = DEFAULT_SHARED_TYPE
    ) where {I, H <: Set{Tuple{I, I}}, S <: SharingBehavior}
    G = Set{I}
    return HessianTracer{I, G, H, S}
end

const DEFAULT_GRADIENT_TRACER = gradient_tracer_type()
const DEFAULT_HESSIAN_TRACER = hessian_tracer_type()

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
struct TracerSparsityDetector{TG <: GradientTracer, TH <: HessianTracer} <:
    ADTypes.AbstractSparsityDetector end

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
struct TracerLocalSparsityDetector{TG <: GradientTracer, TH <: HessianTracer} <:
    ADTypes.AbstractSparsityDetector end

#==========================#
# Convenience constructors #
#==========================#

for D in (:TracerSparsityDetector, :TracerLocalSparsityDetector)
    # Specity both tracer types as arguments
    @eval ($D)(::Type{TG}, ::Type{TH}) where {TG <: GradientTracer, TH <: HessianTracer} = ($D){TG, TH}()

    # Only specify one tracer type, fall back to default for other
    @eval ($D)(::Type{TG}) where {TG <: GradientTracer} = ($D){TG, DEFAULT_HESSIAN_TRACER}()
    @eval ($D)(::Type{TH}) where {TH <: HessianTracer} = ($D){DEFAULT_GRADIENT_TRACER, TH}()

    # Convenience constructor: Only provide pattern types
    # TODO: document
    @eval function ($D)(;
            gradient_pattern_type::Type{G} = DEFAULT_SET_TYPE,
            hessian_pattern_type::Type{H} = DEFAULT_DICT_TYPE,
            shared_hessian::Type{S} = DEFAULT_SHARED_TYPE,
        ) where {G, H, S}
        TG = gradient_tracer_type(G)
        TH = hessian_tracer_type(H, S)
        return ($D)(TG, TH)
    end

    # Pretty printing
    @eval function Base.show(io::IO, d::$D{TG, TH}) where {TG, TH}
        if TG == DEFAULT_GRADIENT_TRACER && TH == DEFAULT_HESSIAN_TRACER
            print(io, $D, "()")
        else
            print(io, $D, "{", TG, ",", TH, "}()")
        end
        return nothing
    end
end

#===============#
# ADTypes calls #
#===============#

# Global
function ADTypes.jacobian_sparsity(f, x, ::TracerSparsityDetector{TG, TH}) where {TG, TH}
    return _jacobian_sparsity(f, x, TG)
end

function ADTypes.jacobian_sparsity(f!, y, x, ::TracerSparsityDetector{TG, TH}) where {TG, TH}
    return _jacobian_sparsity(f!, y, x, TG)
end

function ADTypes.hessian_sparsity(f, x, ::TracerSparsityDetector{TG, TH}) where {TG, TH}
    return _hessian_sparsity(f, x, TH)
end

# Local
function ADTypes.jacobian_sparsity(f, x, ::TracerLocalSparsityDetector{TG, TH}) where {TG, TH}
    return _local_jacobian_sparsity(f, x, TG)
end

function ADTypes.jacobian_sparsity(
        f!, y, x, ::TracerLocalSparsityDetector{TG, TH}
    ) where {TG, TH}
    return _local_jacobian_sparsity(f!, y, x, TG)
end

function ADTypes.hessian_sparsity(f, x, ::TracerLocalSparsityDetector{TG, TH}) where {TG, TH}
    return _local_hessian_sparsity(f, x, TH)
end


#============#
# Allocation #
#============#

# Stable API to allow packages like DI to allocate caches of tracers

"""
    jacobian_eltype(x, detector)

Act like `eltype(x)` but return the matching number type used inside Jacobian sparsity detection.
"""
jacobian_eltype(x, ::TracerSparsityDetector{TG}) where {TG} = TG
function jacobian_eltype(x, ::TracerLocalSparsityDetector{TG}) where {TG}
    return Dual{eltype(x), TG}
end

"""
    hessian_eltype(x, detector)

Act like `eltype(x)` but return the matching number type used inside Hessian sparsity detection.
"""
hessian_eltype(x, ::TracerSparsityDetector{TG, TH}) where {TG, TH} = TH
function hessian_eltype(x, ::TracerLocalSparsityDetector{TG, TH}) where {TG, TH}
    return Dual{eltype(x), TH}
end

"""
    jacobian_buffer(x, detector)

Allocate a buffer similiar to `x` with the required tracer type for Jacobian sparsity detection.
Thin wrapper around `similar` that doesn't expose internal types.
"""
function jacobian_buffer(
        x, detector::Union{TracerSparsityDetector, TracerLocalSparsityDetector}
    )
    return similar(x, jacobian_eltype(x, detector))
end

"""
    hessian_buffer(x, detector)

Allocate a buffer similiar to `x` with the required tracer type for Hessian sparsity detection.
Thin wrapper around `similar` that doesn't expose internal types.
"""
function hessian_buffer(
        x, detector::Union{TracerSparsityDetector, TracerLocalSparsityDetector}
    )
    return similar(x, hessian_eltype(x, detector))
end
