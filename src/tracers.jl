abstract type AbstractTracer <: Real end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)
empty(T) = T()

sparse_vector(T, index) = T([index])

#================#
# Elementwise OR #
#================#

function ×(::Type{H}, a::G, b::G) where {I,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}}}
    return H((i, j) for i in a, j in b)
end

×(a::G, b::G) where {I,G<:AbstractSet{I}} = ×(Set{Tuple{I,I}}, a, b)

#====================#
# ConnectivityTracer #
#====================#

"""
$(TYPEDEF)

`Real` number type keeping track of input indices of previous computations.

For a higher-level interface, refer to [`connectivity_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> inputs = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> SparseConnectivityTracer.ConnectivityTracer(inputs)
SparseConnectivityTracer.ConnectivityTracer{Set{Int64}}(1, 3)
```
"""
struct ConnectivityTracer{C<:AbstractSet{<:Integer}} <: AbstractTracer
    "Sparse binary vector representing non-zero indices of connected, enumerated inputs."
    inputs::C
end

inputs(t::ConnectivityTracer) = t.inputs

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort(collect(inputs(t)))), "$(typeof(t))(", ',', ')', true
    )
end

function empty(::Type{ConnectivityTracer{C}}) where {C}
    return ConnectivityTracer{C}(empty(C))
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
function ConnectivityTracer{C}(::Real) where {C<:AbstractSet{<:Integer}}
    return empty(ConnectivityTracer{C})
end

ConnectivityTracer{C}(t::ConnectivityTracer{C}) where {C<:AbstractSet{<:Integer}} = t
ConnectivityTracer(t::ConnectivityTracer) = t

#================#
# GradientTracer #
#================#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient entries.

For a higher-level interface, refer to [`jacobian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> grad = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> SparseConnectivityTracer.GradientTracer(grad)
SparseConnectivityTracer.GradientTracer{Set{Int64}}(1, 3)
```
"""
struct GradientTracer{G<:AbstractSet{<:Integer}} <: AbstractTracer
    "Sparse binary vector representing non-zero entries in the gradient."
    grad::G
end

gradient(t::GradientTracer) = t.grad

function Base.show(io::IO, t::GradientTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort(collect(gradient(t)))), "$(typeof(t))(", ',', ')', true
    )
end

function empty(::Type{GradientTracer{G}}) where {G}
    return GradientTracer{G}(empty(G))
end

function GradientTracer{G}(::Real) where {G<:AbstractSet{<:Integer}}
    return empty(GradientTracer{G})
end

GradientTracer{G}(t::GradientTracer{G}) where {G<:AbstractSet{<:Integer}} = t
GradientTracer(t::GradientTracer) = t

#===============#
# HessianTracer #
#===============#

"""
$(TYPEDEF)

`Real` number type keeping track of non-zero gradient and Hessian entries.

For a higher-level interface, refer to [`hessian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> grad = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> hess = Set([(1, 1), (2, 3), (3, 2)])
Set{Tuple{Int64, Int64}} with 3 elements:
  (3, 2)
  (1, 1)
  (2, 3)

julia> SparseConnectivityTracer.HessianTracer(grad, hess)
SparseConnectivityTracer.HessianTracer{Set{Int64}, Set{Tuple{Int64, Int64}}}(
  Gradient: Set([3, 1]),
  Hessian:  Set([(3, 2), (1, 1), (2, 3)])
)
```
"""
struct HessianTracer{G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}} <:
       AbstractTracer
    "Sparse binary vector representation of non-zero entries in the gradient."
    grad::G
    "Sparse binary matrix representation of non-zero entries in the Hessian."
    hess::H
end

gradient(t::HessianTracer) = t.grad
hessian(t::HessianTracer)  = t.hess

function Base.show(io::IO, t::HessianTracer)
    println(io, "$(eltype(t))(")
    println(io, "  Gradient: ", gradient(t), ",")
    println(io, "  Hessian:  ", hessian(t))
    print(io, ")")
    return nothing
end

function empty(::Type{HessianTracer{G,H}}) where {G,H}
    return HessianTracer{G,H}(empty(G), empty(H))
end

function HessianTracer{G,H}(
    ::Real
) where {G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}}
    return empty(HessianTracer{G,H})
end

function HessianTracer{G,H}(
    t::HessianTracer{G,H}
) where {G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}}
    return t
end
HessianTracer(t::HessianTracer) = t

#================================#
# Dual numbers for local tracing #
#================================#

"""
$(TYPEDEF)

Dual `Real` number type keeping track of the results of a primal computation as well as a tracer.

## Fields
$(TYPEDFIELDS)
"""
struct Dual{P<:Real,T<:Union{ConnectivityTracer,GradientTracer,HessianTracer}} <:
       AbstractTracer
    primal::P
    tracer::T

    function Dual{P,T}(
        primal::P, tracer::T
    ) where {P<:Number,T<:Union{ConnectivityTracer,GradientTracer,HessianTracer}}
        if P <: AbstractTracer
            error("Primal value of Dual tracer can't be an AbstractTracer.")
        end
        return new{P,T}(primal, tracer)
    end
end
Dual(primal::P, tracer::T) where {P,T} = Dual{P,T}(primal, tracer)

# TODO: support ConnectivityTracer

primal(d::Dual) = d.primal
tracer(d::Dual) = d.tracer

inputs(d::Dual{P,T}) where {P,T<:ConnectivityTracer} = inputs(d.tracer)
gradient(d::Dual{P,T}) where {P,T<:GradientTracer} = gradient(d.tracer)
gradient(d::Dual{P,T}) where {P,T<:HessianTracer} = gradient(d.tracer)
hessian(d::Dual{P,T}) where {P,T<:HessianTracer} = hessian(d.tracer)

function Dual{P,T}(x::Real) where {P<:Real,T<:AbstractTracer}
    return Dual(convert(P, x), empty(T))
end

#===========#
# Utilities #
#===========#

"""
    create_tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
function create_tracer(::Type{Dual{P,T}}, primal::Real, index::Integer) where {P,T}
    return Dual(primal, create_tracer(T, primal, index))
end

function create_tracer(::Type{GradientTracer{G}}, ::Real, index::Integer) where {G}
    return GradientTracer{G}(sparse_vector(G, index))
end
function create_tracer(::Type{ConnectivityTracer{C}}, ::Real, index::Integer) where {C}
    return ConnectivityTracer{C}(sparse_vector(C, index))
end
function create_tracer(::Type{HessianTracer{G,H}}, ::Real, index::Integer) where {G,H}
    return HessianTracer{G,H}(sparse_vector(G, index), empty(H))
end
