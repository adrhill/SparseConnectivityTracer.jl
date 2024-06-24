abstract type AbstractTracer <: Real end

#===================#
# Set operations    #
#===================#

myempty(::Type{S}) where {S<:AbstractSet} = S()
seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S(i)

product(a::AbstractSet{I}, b::AbstractSet{I}) where {I} = Set((i, j) for i in a, j in b)

function union_product!(
    h::H, gx::G, gy::G
) where {I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}}}
    hxy = product(gx, gy)
    return union!(h, hxy)
end

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
struct ConnectivityTracer{I} <: AbstractTracer
    "Sparse representation of connected inputs."
    inputs::I
    "Indicator whether pattern in tracer contains only zeros."
    isempty::Bool

    function ConnectivityTracer{I}(inputs::I, isempty::Bool=false) where {I}
        return new{I}(inputs, isempty)
    end
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{I}(::Real) where {I} = myempty(ConnectivityTracer{I})
ConnectivityTracer{I}(t::ConnectivityTracer{I}) where {I} = t
ConnectivityTracer(t::ConnectivityTracer) = t

inputs(t::ConnectivityTracer) = t.inputs
isemptytracer(t::ConnectivityTracer) = t.isempty

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
struct GradientTracer{G} <: AbstractTracer
    "Sparse representation of non-zero entries in the gradient."
    gradient::G
    "Indicator whether gradient in tracer contains only zeros."
    isempty::Bool

    function GradientTracer{G}(gradient::G, isempty::Bool=false) where {G}
        return new{G}(gradient, isempty)
    end
end

GradientTracer{G}(::Real) where {G} = myempty(GradientTracer{G})
GradientTracer{G}(t::GradientTracer{G}) where {G} = t
GradientTracer(t::GradientTracer) = t

gradient(t::GradientTracer) = t.gradient
isemptytracer(t::GradientTracer) = t.isempty

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

The last type parameter `shared` is a `Bool` indicating whether the Hessian should be shared among all intermediate scalar quantities.

## Fields
$(TYPEDFIELDS)
"""
struct HessianTracer{G,H,shared} <: AbstractTracer
    "Sparse representation of non-zero entries in the gradient and the Hessian."
    gradient::G
    "Sparse representation of non-zero entries in the Hessian."
    hessian::H
    "Indicator whether gradient and Hessian in tracer both contain only zeros."
    isempty::Bool

    function HessianTracer{G,H,shared}(
        gradient::G, hessian::H, isempty::Bool=false
    ) where {G,H,shared}
        return new{G,H,shared}(gradient, hessian, isempty)
    end
end

HessianTracer{G,H,shared}(::Real) where {G,H,shared} = myempty(HessianTracer{G,H,shared})
HessianTracer{G,H,shared}(t::HessianTracer{G,H,shared}) where {G,H,shared} = t
HessianTracer(t::HessianTracer) = t

gradient(t::HessianTracer) = t.gradient
hessian(t::HessianTracer) = t.hessian
isemptytracer(t::HessianTracer) = t.isempty

isshared(::Type{HessianTracer{G,H,shared}}) where {G,H,shared} = shared

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

myempty(::Type{ConnectivityTracer{I}}) where {I} = ConnectivityTracer{I}(myempty(I), true)
myempty(::Type{GradientTracer{G}}) where {G} = GradientTracer{G}(myempty(G), true)

function myempty(::Type{HessianTracer{G,H,shared}}) where {G,H,shared}
    return HessianTracer{G,H,shared}(myempty(G), myempty(H), true)
end

"""
    create_tracer(T, index)

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref) from a single input and its index
"""
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
