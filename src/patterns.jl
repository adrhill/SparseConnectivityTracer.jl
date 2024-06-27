"""
    AbstractPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractPattern
├── AbstractGradientPattern: used in GradientTracer
│   └── IndexSetGradientPattern
└── AbstractHessianPattern: used in HessianTracer
    ├── IndexSetHessianPattern
    └── SharedIndexSetHessianPattern
```
"""
abstract type AbstractPattern end

"""
    isshared(pattern)

Indicates whether patterns share memory (mutate).
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
product(a::AbstractSet{I}, b::AbstractSet{I}) where {I<:Integer} =
    Set((i, j) for i in a, j in b)

function union_product!(
    hessian::SH, gradient_x::SG, gradient_y::SG
) where {I<:Integer,SG<:AbstractSet{I},SH<:AbstractSet{Tuple{I,I}}}
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
"""
struct IndexSetHessianPattern{
    I<:Integer,SG<:AbstractSet{I},SH<:AbstractSet{Tuple{I,I}},mutating
} <: AbstractHessianPattern
    gradient::SG
    hessian::SH
end
isshared(::Type{IndexSetHessianPattern{I,SG,SH,true}}) where {I,SG,SH} = true

function myempty(::Type{P}) where {I,SG,SH,M,P<:IndexSetHessianPattern{I,SG,SH,M}}
    return P(myempty(SG), myempty(SH))
end
function create_patterns(
    ::Type{P}, xs, is
) where {I,SG,SH,M,P<:IndexSetHessianPattern{I,SG,SH,M}}
    gradients = seed.(SG, is)
    hessian = myempty(SH)
    return P.(gradients, Ref(hessian))
end

# Tracer compatibility
gradient(s::IndexSetHessianPattern) = s.gradient
hessian(s::IndexSetHessianPattern) = s.hessian
