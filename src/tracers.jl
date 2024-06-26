abstract type AbstractTracer{P<:AbstractPattern} <: Real end

#====================#
# ConnectivityTracer #
#====================#

"""
$(TYPEDEF)

`Real` number type keeping track of input indices of previous computations.

For a higher-level interface, refer to [`connectivity_pattern`](@ref).

## Fields
$(TYPEDFIELDS)
"""
struct ConnectivityTracer{P<:AbstractVectorPattern} <: AbstractTracer{P}
    "Sparse representation of connected inputs."
    pattern::P
    "Indicator whether pattern in tracer contains only zeros."
    isempty::Bool

    function ConnectivityTracer{P}(inputs::P, isempty::Bool=false) where {P}
        return new{P}(inputs, isempty)
    end
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{P}(::Real) where {P} = myempty(ConnectivityTracer{P})
ConnectivityTracer{P}(t::ConnectivityTracer{P}) where {P} = t
ConnectivityTracer(t::ConnectivityTracer) = t

isemptytracer(t::ConnectivityTracer) = t.isempty
pattern(t::ConnectivityTracer) = t.pattern
inputs(t::ConnectivityTracer) = inputs(pattern(t))

function Base.show(io::IO, t::ConnectivityTracer)
    print(io, typeof(t))
    if isemptytracer(t)
        print(io, "()")
    else
        printsorted(io, inputs(t))
    end
    println(io)
    return nothing
end

#================#
# GradientTracer #
#================#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient entries.

For a higher-level interface, refer to [`jacobian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)
"""
struct GradientTracer{P<:AbstractVectorPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient."
    pattern::P
    "Indicator whether gradient in tracer contains only zeros."
    isempty::Bool

    function GradientTracer{P}(gradient::P, isempty::Bool=false) where {P}
        return new{P}(gradient, isempty)
    end
end

GradientTracer{P}(::Real) where {P} = myempty(GradientTracer{P})
GradientTracer{P}(t::GradientTracer{P}) where {P} = t
GradientTracer(t::GradientTracer) = t

isemptytracer(t::GradientTracer) = t.isempty
pattern(t::GradientTracer) = t.pattern
gradient(t::GradientTracer) = gradient(pattern(t))

function Base.show(io::IO, t::GradientTracer)
    print(io, typeof(t))
    if isemptytracer(t)
        print(io, "()")
    else
        printsorted(io, gradient(t))
    end
    println(io)
    return nothing
end

#===============#
# HessianTracer #
#===============#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient and Hessian entries.

For a higher-level interface, refer to [`hessian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)
"""
struct HessianTracer{P<:AbstractHessianPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient and the Hessian."
    pattern::P
    "Indicator whether gradient and Hessian in tracer both contain only zeros."
    isempty::Bool

    function HessianTracer{P}(pattern::P, isempty::Bool=false) where {P}
        return new{P}(pattern, isempty)
    end
end

HessianTracer{P}(::Real) where {P} = myempty(HessianTracer{P})
HessianTracer{P}(t::HessianTracer{P}) where {P} = t
HessianTracer(t::HessianTracer) = t

isemptytracer(t::HessianTracer) = t.isempty
pattern(t::HessianTracer) = t.pattern
gradient(t::HessianTracer) = gradient(pattern(t))
hessian(t::HessianTracer) = hessian(pattern(t))

function Base.show(io::IO, t::HessianTracer)
    print(io, typeof(t))
    if isemptytracer(t)
        print(io, "()")
    else
        print(io, "(\n", "  Gradient:")
        printlnsorted(io, gradient(t))
        print(io, "  Hessian: ")
        printlnsorted(io, hessian(t))
        println(io, ")")
    end
    return nothing
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
struct Dual{P<:Real,T<:AbstractTracer} <: Real
    primal::P
    tracer::T

    function Dual{P,T}(primal::P, tracer::T) where {P<:Number,T<:AbstractTracer}
        if P <: AbstractTracer || P <: Dual
            error("Primal value of Dual tracer can't be an AbstractTracer.")
        end
        return new{P,T}(primal, tracer)
    end
end

primal(d::Dual) = d.primal
tracer(d::Dual) = d.tracer

inputs(d::Dual{P,T}) where {P,T<:ConnectivityTracer} = inputs(tracer(d))
gradient(d::Dual{P,T}) where {P,T<:GradientTracer}   = gradient(tracer(d))
gradient(d::Dual{P,T}) where {P,T<:HessianTracer}    = gradient(tracer(d))
hessian(d::Dual{P,T}) where {P,T<:HessianTracer}     = hessian(tracer(d))
isemptytracer(d::Dual)                               = isemptytracer(tracer(d))

Dual{P,T}(d::Dual{P,T}) where {P<:Real,T<:AbstractTracer} = d
Dual(primal::P, tracer::T) where {P,T} = Dual{P,T}(primal, tracer)

function Dual{P,T}(x::Real) where {P<:Real,T<:AbstractTracer}
    return Dual(convert(P, x), myempty(T))
end

#===========#
# Utilities #
#===========#

"""
  myempty(T)
  myempty(tracer)
  myempty(pattern)


Constructor for an empty tracer or pattern of type `T` representing a new number (usually an empty pattern).
"""
myempty(::T) where {T<:AbstractTracer} = myempty(T)

# myempty(::Type{T}) where {P,T<:AbstractTracer{P}}   = T(myempty(P), true) # JET complains about this
myempty(::Type{T}) where {P,T<:ConnectivityTracer{P}} = T(myempty(P), true)
myempty(::Type{T}) where {P,T<:GradientTracer{P}}     = T(myempty(P), true)
myempty(::Type{T}) where {P,T<:HessianTracer{P}}      = T(myempty(P), true)

"""
  seed(T, i)
  seed(tracer, i)
  seed(pattern, i)

Constructor for a tracer or pattern of type `T` that only contains the given index `i`.
"""
seed(::T, i) where {T<:AbstractTracer} = seed(T, i)

# seed(::Type{T}, i) where {P,T<:AbstractTracer{P}}   = T(seed(P, i)) # JET complains about this
seed(::Type{T}, i) where {P,T<:ConnectivityTracer{P}} = T(seed(P, i))
seed(::Type{T}, i) where {P,T<:GradientTracer{P}}     = T(seed(P, i))
seed(::Type{T}, i) where {P,T<:HessianTracer{P}}      = T(seed(P, i))

"""
    create_tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
function create_tracer(::Type{T}, ::Real, index::Integer) where {P,T<:AbstractTracer{P}}
    return T(seed(P, index))
end

function create_tracer(::Type{Dual{P,T}}, primal::Real, index::Integer) where {P,T}
    return Dual(primal, create_tracer(T, primal, index))
end

# Pretty-printing of Dual tracers
name(::Type{T}) where {T<:ConnectivityTracer} = "ConnectivityTracer"
name(::Type{T}) where {T<:GradientTracer}     = "GradientTracer"
name(::Type{T}) where {T<:HessianTracer}      = "HessianTracer"
name(::Type{D}) where {P,T,D<:Dual{P,T}}      = "Dual-$(name(T))"
name(::T) where {T<:AbstractTracer}           = name(T)
name(::D) where {D<:Dual}                     = name(D)

# Utilities for printing sets
printsorted(io::IO, x) = Base.show_delim_array(io, sort(x), "(", ',', ')', true)
printsorted(io::IO, s::AbstractSet) = printsorted(io, collect(s))
function printlnsorted(io::IO, x)
    printsorted(io, x)
    println(io)
    return nothing
end
