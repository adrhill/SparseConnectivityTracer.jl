"""
    AbstractPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractPattern
├── AbstractVectorPattern: used in GradientTracer, ConnectivityTracer
│   └── IndexSetVectorPattern
└── AbstractHessianPattern: used in HessianTracer
    └── IndexSetHessianPattern
```
"""
abstract type AbstractPattern end

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
struct IndexSetVectorPattern{I<:Integer,S<:AbstractSet{I}} <: AbstractVectorPattern
    "Set of indices represting non-zero entries ``i`` in a vector."
    vector::S
end

set(v::IndexSetVectorPattern) = v.vector

Base.show(io::IO, s::IndexSetVectorPattern) = Base.show(io, s.vector)

function myempty(::Type{IndexSetVectorPattern{I,S}}) where {I,S}
    return IndexSetVectorPattern{I,S}(myempty(S))
end
function seed(::Type{IndexSetVectorPattern{I,S}}, i) where {I,S}
    return IndexSetVectorPattern{I,S}(seed(S, i))
end

# Tracer compatibility
inputs(s::IndexSetVectorPattern) = s.vector
gradient(s::IndexSetVectorPattern) = s.vector

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
    IndexSetHessianPattern(vector::AbstractVectorPattern, mat::AbstractMatrixPattern)

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
