"""
    AbstractPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractPattern
├── AbstractGradientPattern: used in GradientTracer
│   └── IndexSetGradientPattern
└── AbstractHessianPattern: used in HessianTracer
    └── IndexSetHessianPattern
```
"""
abstract type AbstractPattern end

"""
    isshared(pattern)

Indicates whether patterns **always** share memory and whether operators are **allowed** to mutate their `AbstractTracer` arguments. 

If `false`, patterns **can** share memory and operators are **prohibited** from mutating `AbstractTracer` arguments.

## Note
In practice, memory sharing is limited to second-order information in `AbstractHessianPattern`.
"""
isshared(::P) where {P<:AbstractPattern} = isshared(P)
isshared(::Type{P}) where {P<:AbstractPattern} = false

"""
  myempty(T)
  myempty(tracer)
  myempty(pattern)


Constructor for an empty tracer or pattern of type `T` representing a new number (usually an empty pattern).
"""
myempty

"""
    create_patterns(P, xs, is)

Convenience constructor for patterns of type `P` for multiple inputs `xs` and their indices `is`.
"""
create_patterns

#==========================#
# Utilities on AbstractSet #
#==========================#

myempty(::Type{S}) where {S<:AbstractSet} = S()
seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S(i)

""""
    product(a::S{T}, b::S{T})::S{Tuple{T,T}}

Inner product of set-like inputs `a` and `b`.
"""
function product(a::AbstractSet{I}, b::AbstractSet{I}) where {I<:Integer}
    # Since we return `Symmetric` Hessian, we only have to keep track of index-tuples (i,j) with i≤j.
    return Set((i, j) for i in a, j in b if i <= j)
end

function union_product!(
    hessian::H, gradient_x::G, gradient_y::G
) where {I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}}}
    hxy = product(gradient_x, gradient_y)
    return union!(hessian, hxy)
end

#=======================#
# AbstractGradientPattern #
#=======================#

# For use with GradientTracer.

"""
    AbstractGradientPattern <: AbstractPattern

Abstract supertype of sparsity patterns representing a vector.
For use with [`GradientTracer`](@ref).

## Expected interface

* [`myempty`](@ref)
* [`create_patterns`](@ref)
* `gradient(p::MyPattern)`: return non-zero indices `i` in the gradient representation
* [`isshared`](@ref) in case the pattern is shared (mutates). Defaults to false.
"""
abstract type AbstractGradientPattern <: AbstractPattern end

"""
$(TYPEDEF)

Gradient sparsity pattern represented by an `AbstractSet` of indices ``{i}`` of non-zero values.

## Fields
$(TYPEDFIELDS)
"""
struct IndexSetGradientPattern{I<:Integer,S<:AbstractSet{I}} <: AbstractGradientPattern
    "Set of indices represting non-zero entries ``i`` in a vector."
    gradient::S
end

set(v::IndexSetGradientPattern) = v.gradient

Base.show(io::IO, p::IndexSetGradientPattern) = Base.show(io, set(p))

function myempty(::Type{IndexSetGradientPattern{I,S}}) where {I,S}
    return IndexSetGradientPattern{I,S}(myempty(S))
end
function create_patterns(::Type{P}, xs, is) where {I,S,P<:IndexSetGradientPattern{I,S}}
    sets = seed.(S, is)
    return P.(sets)
end

# Tracer compatibility
gradient(s::IndexSetGradientPattern) = s.gradient

#========================#
# AbstractHessianPattern #
#========================#

# For use with HessianTracer.

"""
    AbstractHessianPattern <: AbstractPattern

Abstract supertype of sparsity patterns representing both gradient and Hessian sparsity.
For use with [`HessianTracer`](@ref).

## Expected interface

* [`myempty`](@ref)
* [`create_patterns`](@ref)
* `gradient(p::MyPattern)`: return non-zero indices `i` in the first-order representation
* `hessian(p::MyPattern)`: return non-zero indices `(i, j)` in the second-order representation
* [`isshared`](@ref) in case the pattern is shared (mutates). Defaults to false.
"""
abstract type AbstractHessianPattern <: AbstractPattern end

"""
$(TYPEDEF)

Hessian sparsity pattern represented by:
* an `AbstractSet` of indices ``i`` of non-zero values representing first-order sparsity
* an `AbstractSet` of index tuples ``(i,j)`` of non-zero values representing second-order sparsity

## Fields
$(TYPEDFIELDS)

## Internals

The last type parameter `shared` is a `Bool` indicating whether the `hessian` field of this object should be shared among all intermediate scalar quantities involved in a function.
"""
struct IndexSetHessianPattern{
    I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}},shared
} <: AbstractHessianPattern
    gradient::G
    hessian::H
end
isshared(::Type{IndexSetHessianPattern{I,G,H,true}}) where {I,G,H} = true

function myempty(::Type{P}) where {I,G,H,S,P<:IndexSetHessianPattern{I,G,H,S}}
    return P(myempty(G), myempty(H))
end
function create_patterns(
    ::Type{P}, xs, is
) where {I,G,H,S,P<:IndexSetHessianPattern{I,G,H,S}}
    gradients = seed.(G, is)
    hessian = myempty(H)
    # Even if `shared=false`, sharing a single reference to `hessian` is allowed upon initialization, 
    # since mutation is prohibited when `isshared` is false.
    return P.(gradients, Ref(hessian))
end

# Tracer compatibility
gradient(s::IndexSetHessianPattern) = s.gradient
hessian(s::IndexSetHessianPattern) = s.hessian
