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
abstract type AbstractTracer{P <: AbstractPattern} <: Real end

#================#
# GradientTracer #
#================#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient entries.

## Fields
$(TYPEDFIELDS)
"""
struct GradientTracer{P <: AbstractGradientPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient."
    pattern::P
    "Indicator whether gradient in tracer contains only zeros."
    isempty::Bool

    function GradientTracer{P}(gradient::P, isempty::Bool = false) where {P}
        return new{P}(gradient, isempty)
    end
end

GradientTracer{P}(::Real) where {P} = myempty(GradientTracer{P})
GradientTracer{P}(t::GradientTracer{P}) where {P} = t

isemptytracer(t::GradientTracer) = t.isempty
pattern(t::GradientTracer) = t.pattern
gradient(t::GradientTracer) = gradient(pattern(t))

#===============#
# HessianTracer #
#===============#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient and Hessian entries.

## Fields
$(TYPEDFIELDS)
"""
struct HessianTracer{P <: AbstractHessianPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient and the Hessian."
    pattern::P
    "Indicator whether gradient and Hessian in tracer both contain only zeros."
    isempty::Bool

    function HessianTracer{P}(pattern::P, isempty::Bool = false) where {P}
        return new{P}(pattern, isempty)
    end
end

HessianTracer{P}(::Real) where {P} = myempty(HessianTracer{P})
HessianTracer{P}(t::HessianTracer{P}) where {P} = t

isemptytracer(t::HessianTracer) = t.isempty
pattern(t::HessianTracer) = t.pattern
gradient(t::HessianTracer) = gradient(pattern(t))
hessian(t::HessianTracer) = hessian(pattern(t))

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

shared(::Type{T}) where {P, T <: HessianTracer{P}} = shared(P)

myempty(::Type{GradientTracer{P}}) where {P} = GradientTracer{P}(myempty(P), true)
myempty(::Type{HessianTracer{P}}) where {P} = HessianTracer{P}(myempty(P), true)

"""
    create_tracers(T, xs, indices)

Convenience constructor for [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref) 
from multiple inputs `xs` and their indices `is`.
"""
function create_tracers(
        ::Type{T}, xs::AbstractArray{<:Real, N}, indices::AbstractArray{<:Integer, N}
    ) where {P <: AbstractPattern, T <: AbstractTracer{P}, N}
    patterns = create_patterns(P, xs, indices)
    return T.(patterns)
end

function create_tracers(
        ::Type{D}, xs::AbstractArray{<:Real, N}, indices::AbstractArray{<:Integer, N}
    ) where {P, T, D <: Dual{P, T}, N}
    tracers = create_tracers(T, xs, indices)
    return D.(xs, tracers)
end
