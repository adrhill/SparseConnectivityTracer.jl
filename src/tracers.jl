abstract type AbstractTracer{P<:AbstractPattern} <: Real end

#================#
# GradientTracer #
#================#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient entries.

## Fields
$(TYPEDFIELDS)
"""
struct GradientTracer{P<:AbstractGradientPattern} <: AbstractTracer{P}
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

## Fields
$(TYPEDFIELDS)

## Internals

The last type parameter `shared` is a `Bool` indicating whether the `hessian` field of this object should be shared among all intermediate scalar quantities involved in a function.
It is not yet part of the public API, and users should always set it to `false`.
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

gradient(d::Dual{P,T}) where {P,T<:GradientTracer} = gradient(tracer(d))
gradient(d::Dual{P,T}) where {P,T<:HessianTracer}  = gradient(tracer(d))
hessian(d::Dual{P,T}) where {P,T<:HessianTracer}   = hessian(tracer(d))
isemptytracer(d::Dual)                             = isemptytracer(tracer(d))

Dual{P,T}(d::Dual{P,T}) where {P<:Real,T<:AbstractTracer} = d
Dual(primal::P, tracer::T) where {P,T} = Dual{P,T}(primal, tracer)

function Dual{P,T}(x::Real) where {P<:Real,T<:AbstractTracer}
    return Dual(convert(P, x), myempty(T))
end

#===========#
# Utilities #
#===========#

myempty(::T) where {T<:AbstractTracer} = myempty(T)

# myempty(::Type{T}) where {P,T<:AbstractTracer{P}}   = T(myempty(P), true) # JET complains about this
myempty(::Type{T}) where {P,T<:GradientTracer{P}} = T(myempty(P), true)
myempty(::Type{T}) where {P,T<:HessianTracer{P}}  = T(myempty(P), true)

seed(::T, i) where {T<:AbstractTracer} = seed(T, i)

# seed(::Type{T}, i) where {P,T<:AbstractTracer{P}}   = T(seed(P, i)) # JET complains about this
seed(::Type{T}, i) where {P,T<:GradientTracer{P}} = T(seed(P, i))
seed(::Type{T}, i) where {P,T<:HessianTracer{P}}  = T(seed(P, i))

"""
    create_tracer(T, index)

Convenience constructor for [`GradientTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
function create_tracer(::Type{T}, ::Real, index::Integer) where {P,T<:AbstractTracer{P}}
    return T(seed(P, index))
end

function create_tracer(::Type{Dual{P,T}}, primal::Real, index::Integer) where {P,T}
    return Dual(primal, create_tracer(T, primal, index))
end

function create_tracer(::Type{ConnectivityTracer{I}}, ::Real, index::Integer) where {I}
    return ConnectivityTracer{I}(seed(I, index))
end
function create_tracer(::Type{GradientTracer{G}}, ::Real, index::Integer) where {G}
    return GradientTracer{G}(seed(G, index))
end
function create_tracer(
    ::Type{HessianTracer{G,H,shared}}, ::Real, index::Integer
) where {G,H,shared}
    return HessianTracer{G,H,shared}(seed(G, index), myempty(H))
end

"""
    create_tracers(T, xs, indices)

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref) from multiple inputs and their indices.

Allows the creation of shared tracer fields (sofar only for the Hessian).
"""
function create_tracers(
    ::Type{T}, xs::AbstractArray{<:Real,N}, indices::AbstractArray{<:Integer,N}
) where {T<:Union{AbstractTracer,Dual},N}
    return create_tracer.(T, xs, indices)
end

function create_tracers(
    ::Type{HessianTracer{G,H,true}},
    xs::AbstractArray{<:Real,N},
    indices::AbstractArray{<:Integer,N},
) where {G,H,N}
    sh = myempty(H)  # shared
    return map(indices) do index
        HessianTracer{G,H,true}(seed(G, index), sh)
    end
end

function create_tracers(
    ::Type{Dual{P,HessianTracer{G,H,true}}},
    xs::AbstractArray{<:Real,N},
    indices::AbstractArray{<:Integer,N},
) where {P<:Real,G,H,N}
    sh = myempty(H)  # shared
    return map(xs, indices) do x, index
        Dual(x, HessianTracer{G,H,true}(seed(G, index), sh))
    end
end

# Pretty-printing of Dual tracers
name(::Type{T}) where {T<:GradientTracer} = "GradientTracer"
name(::Type{T}) where {T<:HessianTracer}  = "HessianTracer"
name(::Type{D}) where {P,T,D<:Dual{P,T}}  = "Dual-$(name(T))"
name(::T) where {T<:AbstractTracer}       = name(T)
name(::D) where {D<:Dual}                 = name(D)

# Utilities for printing sets
printsorted(io::IO, x) = Base.show_delim_array(io, sort(x), "(", ',', ')', true)
printsorted(io::IO, s::AbstractSet) = printsorted(io, collect(s))
function printlnsorted(io::IO, x)
    printsorted(io, x)
    println(io)
    return nothing
end
