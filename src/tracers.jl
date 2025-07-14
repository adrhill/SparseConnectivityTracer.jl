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

"""
    gradient(pattern::AbstractTracer)
    
Return a representation of non-zero values ``∇f(x)_{i} ≠ 0`` in the gradient.
"""
gradient

"""
    hessian(pattern::HessianTracer)
    
Return a representation of non-zero values ``∇²f(x)_{ij} ≠ 0`` in the Hessian.
"""
hessian

"""
    myempty(T)
    myempty(tracer::AbstractTracer)
    
Constructor for an empty tracer or pattern of type `T` representing a new number (usually an empty pattern).
"""
myempty

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

abstract type SharingBehavior end

"""
Indicates that patterns **always** share memory and that operators are **allowed** to mutate their `HessianTracer` arguments.
In practice, memory sharing is limited to second-order information in `HessianTracer`.
"""
struct Shared <: SharingBehavior end

"""
Indicates that patterns **can** share memory and operators are **prohibited** from mutating `HessianTracer` arguments.
In practice, memory sharing is limited to second-order information in `HessianTracer`.
"""
struct NotShared <: SharingBehavior end

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

HessianTracer{I, G, H, S}(::Real) where {I, G, H, S} = myempty(HessianTracer{I, G, H, S})
HessianTracer{I, G, H, S}(t::HessianTracer{I, G, H, S}) where {I, G, H, S} = t

isemptytracer(t::HessianTracer) = t.isempty
gradient(t::HessianTracer) = t.gradient
hessian(t::HessianTracer) = t.hessian

myempty(::Type{HessianTracer{I, G, H, S}}) where {I, G, H, S} = HessianTracer{I, G, H, S}(myempty(G), myempty(H), true)

function create_tracers(
        ::Type{T}, xs, is
    ) where {I, G, H, S, T <: HessianTracer{I, G, H, S}}
    gradients = map(Base.Fix1(seed, G), is)
    hessian = myempty(H)
    # Even if `NotShared`, sharing a single reference to `hessian` is allowed upon initialization,
    # since mutation is prohibited when `isshared` is false.
    return T.(gradients, Ref(hessian))
end

"""
    shared(pattern)

Indicates whether patterns **always** share memory and whether operators are **allowed** to mutate their `HessianTracer` arguments.
Returns either the `Shared()` or `NotShared()` trait.

If `NotShared()`, patterns **can** share memory and operators are **prohibited** from mutating `HessianTracer` arguments.

## Note
In practice, memory sharing is limited to second-order information in `HessianTracer`.
"""
shared(::T) where {T <: HessianTracer} = shared(T)
shared(::Type{HessianTracer{I, G, H, S}}) where {I, G, H, S} = S()

isshared(::Shared) = true
isshared(::NotShared) = false
isshared(t) = isshared(shared(t))

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
