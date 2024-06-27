"""
    AbstractPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractPattern
├── AbstractGradientPattern: used in GradientTracer, ConnectivityTracer
│   └── IndexSetGradientPattern
└── AbstractHessianPattern: used in HessianTracer
    └── IndexSetHessianPattern
```
"""
abstract type AbstractPattern end

"""
  myempty(T)
  myempty(tracer)
  myempty(pattern)


Constructor for an empty tracer or pattern of type `T` representing a new number (usually an empty pattern).
"""
myempty

"""
  seed(T, i)
  seed(tracer, i)
  seed(pattern, i)

Constructor for a tracer or pattern of type `T` that only contains the given index `i`.
"""
seed

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

* `myempty(::Type{MyPattern})`: return a pattern representing a new number (usually an empty pattern)
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i`
* `gradient(p::MyPattern)`: return non-zero indices `i` for use with `GradientTracer`

Note that besides their names, the last two functions are usually identical.
"""
abstract type AbstractGradientPattern <: AbstractPattern end

"""
$(TYPEDEF)

Vector sparsity pattern represented by an `AbstractSet` of indices ``{i}`` of non-zero values.

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
function seed(::Type{IndexSetGradientPattern{I,S}}, i) where {I,S}
    return IndexSetGradientPattern{I,S}(seed(S, i))
end

# Tracer compatibility
inputs(s::IndexSetGradientPattern) = s.gradient
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

* `myempty(::Type{MyPattern})`: return a pattern representing a new number (usually an empty pattern)
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i` in the first-order representation
* `gradient(p::MyPattern)`: return non-zero indices `i` in the first-order representation
* `hessian(p::MyPattern)`: return non-zero indices `(i, j)` in the second-order representation
"""
abstract type AbstractHessianPattern <: AbstractPattern end

"""
    IndexSetHessianPattern(vector::AbstractGradientPattern, mat::AbstractMatrixPattern)

Gradient and Hessian sparsity patterns constructed by combining two AbstractSets.
"""
struct IndexSetHessianPattern{I<:Integer,SG<:AbstractSet{I},SH<:AbstractSet{Tuple{I,I}}} <:
       AbstractHessianPattern
    gradient::SG
    hessian::SH
end

function myempty(::Type{IndexSetHessianPattern{I,SG,SH}}) where {I,SG,SH}
    return IndexSetHessianPattern{I,SG,SH}(myempty(SG), myempty(SH))
end
function seed(::Type{IndexSetHessianPattern{I,SG,SH}}, index) where {I,SG,SH}
    return IndexSetHessianPattern{I,SG,SH}(seed(SG, index), myempty(SH))
end

# Tracer compatibility
gradient(s::IndexSetHessianPattern) = s.gradient
hessian(s::IndexSetHessianPattern) = s.hessian
