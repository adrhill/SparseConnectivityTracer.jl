#=
The following hierarchy of sparsity pattern representations is implemented in this file:

AbstractSparsityPattern
├── AbstractVectorPattern
│   └── IndexSetVectorPattern: pattern represented by index sets using set-like datastructures
├── AbstractMatrixPattern
│   └── IndexSetMatrixPattern: pattern represented by index sets using set-like datastructures
└── AbstractVectorAndMatrixPattern
    └── CombinedPattern: combines separate AbstractVectorPattern and AbstractMatrixPattern
=#

"""
    AbstractSparsityPattern

Abstract supertype of all sparsity pattern representations.

## Type hierarchy
```
AbstractSparsityPattern
├── AbstractVectorPattern
├── AbstractMatrixPattern
└── AbstractVectorAndMatrixPattern
```
"""
abstract type AbstractSparsityPattern end

# Utilities on AbstractSet
"""
    myempty(S)

Constructor for an empty set-like data structure of type `S`.
"""
@inline myempty(::Type{S}) where {S<:AbstractSet} = S()

"""
    myempty(S)

Constructor for a set-like data structure of type `S` that only contains the given index `i`.
"""
@inline seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S([i])

""""
    product(a::S{T}, b::S{T})::S{Tuple{T,T}}

Inner product of set-like inputs `a` and `b`.
"""
product(a::AbstractSet{I}, b::AbstractSet{I}) where {I} = Set((i, j) for i in a, j in b)

## Vector

"""
    AbstractVectorPattern <: AbstractSparsityPattern

Abstract supertype of sparsity patterns representing a vector.

## Expected interface

* `myempty(::Type{MyPattern})`: return an empty pattern
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i`
* `Base.show`

And for compatibility with tracers:

| Tracer               | Required accessor function |
|:---------------------|:---------------------------|
| `ConnectivityTracer` | `inputs(p::MyPattern)`     |
| `GradientTracer`     | `gradient(p::MyPattern)`   |
| `HessianTracer`      | `gradient(p::MyPattern)`   | 
"""
abstract type AbstractVectorPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

Vector sparsity pattern represented by an `AbstractSet` of indices ``{i}`` of non-zero values.

## Fields
$(TYPEDFIELDS)
"""
struct IndexSetVectorPattern{I<:Integer,S<:AbstractSet{I}} <: AbstractVectorPattern
    "Set of indices represting non-zero entries ``i``."
    inds::S
end

Base.show(io::IO, s::IndexSetVectorPattern) = Base.show(io, s.inds)

@inline function myempty(::Type{IndexSetVectorPattern{I,S}}) where {I,S}
    return IndexSetVectorPattern{I,S}(myempty(S))
end
@inline function seed(::Type{IndexSetVectorPattern{I,S}}, i) where {I,S}
    return IndexSetVectorPattern{I,S}(seed(S, i))
end

# Tracer compatibility
@inline inputs(s::IndexSetVectorPattern) = s.inds
@inline gradient(s::IndexSetVectorPattern) = s.inds

## Matrix

"""
    AbstractMatrixPattern <: AbstractSparsityPattern

Abstract supertype of sparsity patterns representing a vector.

## Expected interface

* `myempty(::Type{MyPattern})`: return an empty pattern
* `Base.show`

And for compatibility with tracers:

| Tracer               | Required accessor function |
|:---------------------|:---------------------------|
| `HessianTracer`      | `hessian(p::MyPattern)`    | 
"""
abstract type AbstractMatrixPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

Matrix sparsity pattern represented by an `AbstractSet` of indices ``{(i,j)}`` of non-zero values.

## Fields
$(TYPEDFIELDS)
"""
struct IndexSetMatrixPattern{I<:Integer,S<:AbstractSet{Tuple{I,I}}} <: AbstractMatrixPattern
    "Set of index tuples represting non-zero entries ``(i, j)``."
    inds::S
end

Base.show(io::IO, s::IndexSetMatrixPattern) = Base.show(io, s.inds)

@inline function myempty(::Type{IndexSetMatrixPattern{I,S}}) where {I,S}
    return IndexSetMatrixPattern{I,S}(myempty(S))
end

hessian(p::IndexSetMatrixPattern) = p.inds

## Vector and Matrix

"""
    AbstractVectorAndMatrixPattern <: AbstractSparsityPattern

Abstract supertype of sparsity patterns representing both a vector and a matrix.

## Expected interface

* `myempty(::Type{MyPattern})`: return an empty pattern
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i` in the vector
* `Base.show`

And for compatibility with tracers:

| Tracer               | Required accessor function                           |
|:---------------------|:-----------------------------------------------------|
| `ConnectivityTracer` | `inputs(p::MyPattern)`                               |
| `GradientTracer`     | `gradient(p::MyPattern)`                             |
| `HessianTracer`      | `gradient(p::MyPattern)` AND `hessian(p::MyPattern)` | 
"""
abstract type AbstractVectorAndMatrixPattern <: AbstractSparsityPattern end

"""
    CombinedPattern(vec::AbstractVectorPattern, mat::AbstractMatrixPattern)

Vector and matrix sparsity pattern constructed by combining two separate vector and matrix representations.
"""
struct CombinedPattern{V<:AbstractVectorPattern,M<:AbstractMatrixPattern} <:
       AbstractVectorAndMatrixPattern
    vec::V
    mat::M

    # Manually allow a subset of vector- and matrix pattern combinations
    # by adding inner constructors
    function CombinedPattern{V,M}(
        vec::V, mat::M
    ) where {I,V<:IndexSetVectorPattern{I},M<:IndexSetMatrixPattern{I}}
        return new{V,M}(vec, mat)
    end
end

function Base.show(io::IO, s::CombinedPattern)
    println(io, "Vector: ", s.vec)
    println(io, "Matrix: ", s.mat)
    return nothing
end

@inline function myempty(::Type{CombinedPattern{V,M}}) where {V,M}
    return CombinedPattern{V,M}(myempty(V), myempty(M))
end
@inline function seed(::Type{CombinedPattern{V,M}}, index) where {V,M}
    return CombinedPattern{V,M}(seed(V, index), myempty(M))
end

# Tracer compatibility
@inline gradient(s::CombinedPattern) = gradient(s.vec)
@inline hessian(s::CombinedPattern) = hessian(s.mat)
