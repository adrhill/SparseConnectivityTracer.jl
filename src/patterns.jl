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
    shared(pattern)

Indicates whether patterns **always** share memory and whether operators are **allowed** to mutate their `AbstractTracer` arguments.
Returns either the `Shared()` or `NotShared()` trait.

If `NotShared()`, patterns **can** share memory and operators are **prohibited** from mutating `AbstractTracer` arguments.

## Note
In practice, memory sharing is limited to second-order information in `AbstractHessianPattern`.
"""
shared(::P) where {P<:AbstractPattern} = shared(P)
shared(::Type{P}) where {P<:AbstractPattern} = NotShared()

abstract type SharingBehavior end
struct Shared <: SharingBehavior end
struct NotShared <: SharingBehavior end

Base.Bool(::Shared) = true
Base.Bool(::NotShared) = false

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

"""
    gradient(pattern)
    
Return a representation of non-zero values ``∇f(x)_{i} ≠ 0`` in the gradient.
"""
gradient

"""
    hessian(pattern)
    
Return a representation of non-zero values ``∇²f(x)_{ij} ≠ 0`` in the Hessian.
"""
hessian

#===========#
# Utilities #
#===========#

myempty(::S) where {S<:AbstractSet} = S()
myempty(::Type{S}) where {S<:AbstractSet} = S()
seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S(i)

""""
    product(a::S{T}, b::S{T})::S{Tuple{T,T}}

Inner product of set-like inputs `a` and `b`.
"""
function product(a::AbstractSet{I}, b::AbstractSet{I}) where {I<:Integer}
    # Since the Hessian is symmetric, we only have to keep track of index-tuples (i,j) with i≤j.
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
* [`gradient`](@ref)
* [`isshared`](@ref) in case the pattern is shared (mutates). Defaults to false.
"""
abstract type AbstractGradientPattern <: AbstractPattern end

"""
$(TYPEDEF)

Gradient sparsity pattern represented by a set.

## Fields
$(TYPEDFIELDS)
"""
struct IndexSetGradientPattern{I<:Integer,S<:AbstractSet{I}} <: AbstractGradientPattern
    "Set of indices ``i`` of non-zero values ``∇f(x)_i ≠ 0`` in the gradient."
    gradient::S
end

Base.show(io::IO, p::IndexSetGradientPattern) = Base.show(io, gradient(p))

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
* [`gradient`](@ref)
* [`hessian`](@ref)
* [`shared`](@ref) in case the pattern is shared (mutates). Defaults to `NotShared()`.
"""
abstract type AbstractHessianPattern <: AbstractPattern end

"""
$(TYPEDEF)

Hessian sparsity pattern represented by two sets.

## Fields
$(TYPEDFIELDS)

## Internals
The last type parameter `shared` is a `Bool` indicating whether the `hessian` field of this object should be shared among all intermediate scalar quantities involved in a function.
"""
struct IndexSetHessianPattern{
    I<:Integer,G<:AbstractSet{I},H<:AbstractSet{Tuple{I,I}},shared<:SharingBehavior
} <: AbstractHessianPattern
    "Set of indices ``i`` of non-zero values ``∇f(x)_i ≠ 0`` in the gradient."
    gradient::G
    "Set of index-tuples ``(i, j)`` of non-zero values ``∇²f(x)_{ij} ≠ 0`` in the Hessian."
    hessian::H
end
shared(::Type{IndexSetHessianPattern{I,G,H,Shared}}) where {I,G,H}    = Shared()
shared(::Type{IndexSetHessianPattern{I,G,H,NotShared}}) where {I,G,H} = NotShared()

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
