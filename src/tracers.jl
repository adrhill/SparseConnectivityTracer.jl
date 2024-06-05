abstract type AbstractTracer{I<:AbstractSparsityPattern} <: Real end

#===================#
# Set operations    #
#===================#

function clever_union(a::AbstractSet{P}, b::AbstractSet{P}) where {P}
    if isempty(a)
        return b
    elseif isempty(b)
        return a
    else
        return union(a, b)
    end
end

product(a::AbstractSet{I}, b::AbstractSet{I}) where {I} = Set((i, j) for i in a, j in b)

function union_product!(
    sh::SH, sgx::SG, sgy::SG
) where {I,SG<:AbstractSet{I},SH<:AbstractSet{Tuple{I,I}}}
    shxy = product(sgx, sgy)
    return union!(sh, shxy)
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
struct ConnectivityTracer{P<:AbstractFirstOrderPattern} <: AbstractTracer{P}
    "Sparse representation of connected inputs."
    pattern::P
    "Indicator whether pattern in tracer contains only zeros."
    isempty::Bool
end
function ConnectivityTracer{P}(pattern::P) where {P<:AbstractFirstOrderPattern}
    return ConnectivityTracer{P}(pattern, false)
end

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort(collect(inputs(t)))), "$(typeof(t))(", ',', ')', true
    )
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
function ConnectivityTracer{P}(::Real) where {P<:AbstractFirstOrderPattern}
    return myempty(ConnectivityTracer{P})
end

ConnectivityTracer{P}(t::ConnectivityTracer{P}) where {P<:AbstractFirstOrderPattern} = t
ConnectivityTracer(t::ConnectivityTracer) = t

inputs(t::ConnectivityTracer) = inputs(t.pattern)

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
struct GradientTracer{P<:AbstractFirstOrderPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient."
    pattern::P
    "Indicator whether pattern in tracer contains only zeros."
    isempty::Bool
end
function GradientTracer{P}(pattern::P) where {P<:AbstractFirstOrderPattern}
    return GradientTracer{P}(pattern, false)
end

function Base.show(io::IO, t::GradientTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort(collect(gradient(t)))), "$(typeof(t))(", ',', ')', true
    )
end

GradientTracer{P}(::Real) where {P<:AbstractFirstOrderPattern} = myempty(GradientTracer{P})
GradientTracer{P}(t::GradientTracer{P}) where {P<:AbstractFirstOrderPattern} = t
GradientTracer(t::GradientTracer) = t

gradient(t::GradientTracer) = gradient(t.pattern)

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
struct HessianTracer{P<:AbstractSecondOrderPattern} <: AbstractTracer{P}
    "Sparse representation of non-zero entries in the gradient and the Hessian."
    pattern::P
    "Indicator whether pattern in tracer contains only zeros."
    isempty::Bool
end
function HessianTracer{P}(pattern::P) where {P<:AbstractSecondOrderPattern}
    return HessianTracer{P}(pattern, false)
end

function Base.show(io::IO, t::HessianTracer)
    println(io, "$(eltype(t))(")
    println(io, "  Gradient: ", gradient(t), ",")
    println(io, "  Hessian:  ", hessian(t))
    print(io, ")")
    return nothing
end

HessianTracer{P}(::Real) where {P<:AbstractSecondOrderPattern} = myempty(HessianTracer{P})
HessianTracer{P}(t::HessianTracer{P}) where {P<:AbstractSecondOrderPattern} = t
HessianTracer(t::HessianTracer) = t

gradient(t::HessianTracer) = gradient(t.pattern)
hessian(t::HessianTracer) = hessian(t.pattern)

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
Dual(primal::P, tracer::T) where {P,T} = Dual{P,T}(primal, tracer)

primal(d::Dual) = d.primal
tracer(d::Dual) = d.tracer

inputs(d::Dual{P,T}) where {P,T<:ConnectivityTracer} = inputs(d.tracer)
gradient(d::Dual{P,T}) where {P,T<:GradientTracer} = gradient(d.tracer)
gradient(d::Dual{P,T}) where {P,T<:HessianTracer} = gradient(d.tracer)
hessian(d::Dual{P,T}) where {P,T<:HessianTracer} = hessian(d.tracer)

function Dual{P,T}(x::Real) where {P<:Real,T<:AbstractTracer}
    return Dual(convert(P, x), myempty(T))
end

#===========#
# Utilities #
#===========#

myempty(::Type{ConnectivityTracer{P}}) where {P} = ConnectivityTracer{P}(myempty(P), true)
myempty(::Type{GradientTracer{P}}) where {P} = GradientTracer{P}(myempty(P), true)
myempty(::Type{HessianTracer{P}}) where {P} = HessianTracer{P}(myempty(P), true)

"""
    create_tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
function create_tracer(::Type{Dual{P,T}}, primal::Real, index::Integer) where {P,T}
    return Dual(primal, create_tracer(T, primal, index))
end

function create_tracer(::Type{ConnectivityTracer{P}}, ::Real, index::Integer) where {P}
    return ConnectivityTracer{P}(seed(P, index))
end
function create_tracer(::Type{GradientTracer{P}}, ::Real, index::Integer) where {P}
    return GradientTracer{P}(seed(P, index))
end
function create_tracer(::Type{HessianTracer{P}}, ::Real, index::Integer) where {P}
    return HessianTracer{P}(seed(P, index), false)
end
