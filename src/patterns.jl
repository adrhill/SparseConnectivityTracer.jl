"""
    AbstractPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractPattern
├── AbstractVectorPattern: used in GradientTracer, ConnectivityTracer
│   └── IndexSetVector
└── AbstractHessianPattern: used in HessianTracer
    └── IndexSetHessian
```
"""
AbstractPattern

"""
    myempty(P)

Constructor for an empty pattern of type `P` representing a new number (usually an empty pattern).
"""
myempty(::P) where {P<:AbstractPattern} = myempty(P)
myempty(::T) where {P<:AbstractPattern,T<:AbstractTracer{P}} = T(myempty(P), true)
myempty(::Type{T}) where {P<:AbstractPattern,T<:AbstractTracer{P}} = T(myempty(P), true)

"""
seed(P, i)

Constructor for a pattern of type `P` that only contains the given index `i`.
"""
seed(::P, i) where {P<:AbstractPattern} = seed(P, i)
seed(::T, i) where {P<:AbstractPattern,T<:AbstractTracer{P}} = T(seed(P, i))
seed(::Type{T}, i) where {P<:AbstractPattern,T<:AbstractTracer{P}} = T(seed(P, i))

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
# AbstractVectorPattern #
#=======================#

# For use with ConnectivityTracer and GradientTracer.

"""
    AbstractVectorPattern <: AbstractPattern

Abstract supertype of sparsity patterns representing a vector.
For use with [`ConnectivityTracer`](@ref) and [`GradientTracer`](@ref).

## Expected interface

* `myempty(::Type{MyPattern})`: return a pattern representing a new number (usually an empty pattern)
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i`
* `inputs(p::MyPattern)`: return non-zero indices `i` for use with `ConnectivityTracer`
* `gradient(p::MyPattern)`: return non-zero indices `i` for use with `GradientTracer`

Note that besides their names, the last two functions are usually identical.
"""
abstract type AbstractVectorPattern <: AbstractPattern end

"""
$(TYPEDEF)

Vector sparsity pattern represented by an `AbstractSet` of indices ``{i}`` of non-zero values.

## Fields
$(TYPEDFIELDS)
"""
struct IndexSetVector{I<:Integer,S<:AbstractSet{I}} <: AbstractVectorPattern
    "Set of indices represting non-zero entries ``i`` in a vector."
    vector::S
end

Base.show(io::IO, s::IndexSetVector) = Base.show(io, s.vector)

function myempty(::Type{IndexSetVector{I,S}}) where {I,S}
    return IndexSetVector{I,S}(myempty(S))
end
function seed(::Type{IndexSetVector{I,S}}, i) where {I,S}
    return IndexSetVector{I,S}(seed(S, i))
end

# Tracer compatibility
inputs(s::IndexSetVector) = s.vector
gradient(s::IndexSetVector) = s.vector

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
    IndexSetHessian(vector::AbstractVectorPattern, mat::AbstractMatrixPattern)

Gradient and Hessian sparsity patterns constructed by combining two AbstractSets.
"""
struct IndexSetHessian{I<:Integer,SG<:AbstractSet{I},SH<:AbstractSet{Tuple{I,I}}} <:
       AbstractHessianPattern
    gradient::SG
    hessian::SH
end

function myempty(::Type{IndexSetHessian{I,SG,SH}}) where {I,SG,SH}
    return IndexSetHessian{I,SG,SH}(myempty(SG), myempty(SH))
end
function seed(::Type{IndexSetHessian{I,SG,SH}}, index) where {I,SG,SH}
    return IndexSetHessian{I,SG,SH}(seed(SG, index), myempty(SH))
end

# Tracer compatibility
gradient(s::IndexSetHessian) = s.gradient
hessian(s::IndexSetHessian) = s.hessian
