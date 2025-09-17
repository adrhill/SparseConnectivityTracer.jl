#= This file implements the ADTypes interface for `AbstractSparsityDetector`s =#
const DEFAULT_SET_TYPE = BitSet
const DEFAULT_DICT_TYPE = Dict{eltype(DEFAULT_SET_TYPE), DEFAULT_SET_TYPE}
const DEFAULT_SHARED_TYPE = NotShared

function gradient_tracer_type(
        ::Type{G} = DEFAULT_SET_TYPE
    ) where {G <: AbstractSet}
    I = eltype(G)
    return GradientTracer{I, G}
end

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


const DOC_KWARGS = """# Keyword arguments
- `gradient_pattern_type::Type`: 
  Data structure used for bookkeeping of gradient sparsity patterns, used in `jacobian_sparsity`.
  Supports concrete subtypes of `AbstactSet{<:Integer}`.
  Defaults to `$DEFAULT_SET_TYPE`.
- `hessian_pattern_type::Type`: 
  Data structure used for bookkeeping of Hessian sparsity patterns, used in `hessian_sparsity`.
  Supports concrete subtypes of `AbstractDict{I, AbstractSet{I}}` or `AbstractSet{Tuple{I, I}}`, where `I <: Integer`.
  Defaults to `$DEFAULT_DICT_TYPE`.
- `shared_hessian_pattern::Bool`: 
  Indicate whether second-order information in Hessian sparsity patterns **always** shares memory and whether operators are **allowed** to mutate `HessianTracers`.
  Defaults to `false`.

If support for further pattern representations is needed, please open a feature request:
https://github.com/adrhill/SparseConnectivityTracer.jl/issues
"""

"""
    TracerSparsityDetector()

Global sparsity detection over the entire input domain using SparseConnectivityTracer.jl. 
For use with [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s `AbstractSparsityDetector` interface.

For local sparsity patterns, use [`TracerLocalSparsityDetector`](@ref).

$DOC_KWARGS

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

## References

- A. Hill and G. Dalle (2025). [*"Sparser, Better, Faster, Stronger: Sparsity Detection for Efficient Automatic Differentiation"*](https://openreview.net/forum?id=GtXSN52nIW)
- A. Hill, G. Dalle, A. Montoison (2025). [*"An Illustrated Guide to Automatic Sparse Differentiation"*](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/)
"""
struct TracerSparsityDetector{TG <: GradientTracer, TH <: HessianTracer} <:
    ADTypes.AbstractSparsityDetector end

"""
    TracerLocalSparsityDetector()

Local sparsity detection using SparseConnectivityTracer.jl. 
For use with [ADTypes.jl](https://github.com/SciML/ADTypes.jl)'s `AbstractSparsityDetector` interface.

For global sparsity patterns, use [`TracerSparsityDetector`](@ref).

$DOC_KWARGS

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

## References

- A. Hill and G. Dalle (2025). [*"Sparser, Better, Faster, Stronger: Sparsity Detection for Efficient Automatic Differentiation"*](https://openreview.net/forum?id=GtXSN52nIW)
- A. Hill, G. Dalle, A. Montoison (2025). [*"An Illustrated Guide to Automatic Sparse Differentiation"*](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/)
"""
struct TracerLocalSparsityDetector{TG <: GradientTracer, TH <: HessianTracer} <:
    ADTypes.AbstractSparsityDetector end

#==========================#
# Convenience constructors #
#==========================#

for D in (:TracerSparsityDetector, :TracerLocalSparsityDetector)
    # Specity both tracer types as arguments (stable, but not public/documented)
    @eval ($D)(::Type{TG}, ::Type{TH}) where {TG <: GradientTracer, TH <: HessianTracer} = ($D){TG, TH}()

    # Only specify one tracer type, fall back to default for other (stable, but not public/documented)
    @eval ($D)(::Type{TG}) where {TG <: GradientTracer} = ($D){TG, DEFAULT_HESSIAN_TRACER}()
    @eval ($D)(::Type{TH}) where {TH <: HessianTracer} = ($D){DEFAULT_GRADIENT_TRACER, TH}()

    # Convenience constructor: Only provide pattern types (public/documented in `DOC_KWARGS`)
    @eval function ($D)(;
            gradient_pattern_type::Type{G} = DEFAULT_SET_TYPE,
            hessian_pattern_type::Type{H} = DEFAULT_DICT_TYPE,
            shared_hessian_pattern::Bool = false,
        ) where {G, H}
        S = shared_hessian_pattern ? Shared : NotShared
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
