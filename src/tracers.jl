"""
    AbstractTracer

Abstract supertype of tracers.

## Type hierarchy
```
AbstractTracer
├── GradientTracer
└── HessianTracer
```

Note that [`Dual`](@ref) is not an `AbstractTracer`.
"""
abstract type AbstractTracer <: Real end

#================#
# GradientTracer #
#================#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient entries.

## Fields
$(TYPEDFIELDS)
"""
struct GradientTracer{I <: Integer, G <: AbstractSet{I}} <: AbstractTracer
    "Set of indices ``i`` of non-zero values ``∇f(x)_i ≠ 0`` in the gradient."
    gradient::G
    "Indicator whether gradient in tracer contains only zeros."
    isempty::Bool

    function GradientTracer{I, G}(gradient::G, isempty::Bool = false) where {I, G}
        return new{I, G}(gradient, isempty)
    end
end

GradientTracer{I, G}(::Real) where {I, G} = myempty(GradientTracer{I, G})
GradientTracer{I, G}(t::GradientTracer{I, G}) where {I, G} = t

isemptytracer(t::GradientTracer) = t.isempty
gradient(t::GradientTracer) = t.gradient

myempty(::Type{GradientTracer{I, G}}) where {I, G} = GradientTracer{I, G}(myempty(G), true)

function create_tracers(::Type{T}, xs, is) where {I, G, T <: GradientTracer{I, G}}
    gradients = map(Base.Fix1(seed, G), is)
    return T.(gradients)
end

#===============#
# HessianTracer #
#===============#

"""
    shared(pattern)

Indicates whether patterns **always** share memory and whether operators are **allowed** to mutate their `AbstractTracer` arguments.
Returns either the `Shared()` or `NotShared()` trait.

If `NotShared()`, patterns **can** share memory and operators are **prohibited** from mutating `AbstractTracer` arguments.

## Note
In practice, memory sharing is limited to second-order information in `HessianTracer`.
"""
shared(::P) where {P <: AbstractPattern} = shared(P)
shared(::Type{P}) where {P <: AbstractPattern} = NotShared()

abstract type SharingBehavior end
struct Shared <: SharingBehavior end
struct NotShared <: SharingBehavior end

isshared(::Shared) = true
isshared(::NotShared) = false

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient and Hessian entries.

## Fields
$(TYPEDFIELDS)
"""
struct HessianTracer{
        I <: Integer, G <: AbstractSet{I}, H <: Union{AbstractDict{I, G}, AbstractSet{Tuple{I, I}}}, S <: SharingBehavior,
    } <: AbstractTracer
    "Set of indices ``i`` of non-zero values ``∇f(x)_i ≠ 0`` in the gradient."
    gradient::G
    "Set of index-tuples ``(i, j)`` of non-zero values ``∇²f(x)_{ij} ≠ 0`` in the Hessian."
    hessian::H
    "Indicator whether gradient and Hessian in tracer both contain only zeros."
    isempty::Bool

    function HessianTracer{I, G, H, S}(gradient::G, hessian::H, isempty::Bool = false) where {I, G, H, S}
        return new{I, G, H, S}(gradient, hessian, isempty)
    end
end

HessianTracer{P}(::Real) where {P} = myempty(HessianTracer{P})
HessianTracer{P}(t::HessianTracer{P}) where {P} = t

isemptytracer(t::HessianTracer) = t.isempty
pattern(t::HessianTracer) = t.pattern
gradient(t::HessianTracer) = t.gradient
hessian(t::HessianTracer) = t.hessian

shared(::Type{HessianTracer{I, G, H, S}}) where {I, G, H, S} = shared(S)

myempty(::Type{HessianTracer{I, G, H}}) where {I, G, H} = HessianTracer{P}(myempty(G), myempty(H), true)

function create_tracers(
        ::Type{T}, xs, is
    ) where {I, G, H, S, T <: HessianTracer{I, G, H, S}}
    gradients = map(Base.Fix1(seed, G), is)
    hessian = myempty(H)
    # Even if `NotShared`, sharing a single reference to `hessian` is allowed upon initialization,
    # since mutation is prohibited when `isshared` is false.
    return T.(gradients, Ref(hessian))
end

#================================#
# Dual numbers for local tracing #
#================================#

"""
$(TYPEDEF)

Dual `Real` number type keeping track of the results of a primal computation as well as a tracer.

## Fields
$(TYPEDFIELDS)
"""
struct Dual{P <: Real, T <: AbstractTracer} <: Real
    primal::P
    tracer::T

    function Dual{P, T}(primal::P, tracer::T) where {P <: Number, T <: AbstractTracer}
        if P <: AbstractTracer || P <: Dual
            error("Primal value of Dual tracer can't be an AbstractTracer.")
        end
        return new{P, T}(primal, tracer)
    end
end

primal(d::Dual) = d.primal
tracer(d::Dual) = d.tracer

gradient(d::Dual{P, T}) where {P, T <: GradientTracer} = gradient(tracer(d))
gradient(d::Dual{P, T}) where {P, T <: HessianTracer} = gradient(tracer(d))
hessian(d::Dual{P, T}) where {P, T <: HessianTracer} = hessian(tracer(d))
isemptytracer(d::Dual) = isemptytracer(tracer(d))

Dual{P, T}(d::Dual{P, T}) where {P <: Real, T <: AbstractTracer} = d
Dual(primal::P, tracer::T) where {P, T} = Dual{P, T}(primal, tracer)

function Dual{P, T}(x::Real) where {P <: Real, T <: AbstractTracer}
    return Dual(convert(P, x), myempty(T))
end

#===========#
# Utilities #
#===========#


"""
    create_tracers(T, xs, indices)

Convenience constructor for [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref) 
from multiple inputs `xs` and their indices `is`.
"""


function create_tracers(
        ::Type{D}, xs::AbstractArray{<:Real, N}, indices::AbstractArray{<:Integer, N}
    ) where {P, T, D <: Dual{P, T}, N}
    tracers = create_tracers(T, xs, indices)
    return D.(xs, tracers)
end
