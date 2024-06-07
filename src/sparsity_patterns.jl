#=
The following hierarchy of sparsity pattern representations is implemented in this file:

AbstractSparsityPattern
├── AbstractVectorPattern
│   └── SimpleVectorIndexSetPattern: pattern represented by index sets using set-like datastructures
├── AbstractMatrixPattern
│   └── SimpleMatrixIndexSetPattern: pattern represented by index sets using set-like datastructures
└── AbstractVectorAndMatrixPattern
    └── CombinedVectorAndMatrixPattern: combines separate AbstractVectorPattern and AbstractMatrixPattern
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
myempty(::Type{S}) where {S<:AbstractSet} = S()

"""
    myempty(S)

Constructor for a set-like data structure of type `S` that only contains the given index `i`.
"""
seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S([i])

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

Vector sparsity pattern represented by an index set ``{i}`` of non-zero values.
Supports set-like datastructures `S`.

## Expected interface

The passed set type `S` has to implement:
* `SparseConnectivityTracer.myempty`
* `SparseConnectivityTracer.seed`
* `SparseConnectivityTracer.product`
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

All of these methods are already implemented for `AbstractSet`s from Julia Base,
which are most commonly used in `SimpleVectorIndexSetPattern`.
Refer to the individual documentation of each function for more information. 

## Fields
$(TYPEDFIELDS)
"""
struct SimpleVectorIndexSetPattern{S} <: AbstractVectorPattern
    "Set of indices represting non-zero entries ``i``."
    inds::S
end
Base.show(io::IO, s::SimpleVectorIndexSetPattern) = Base.show(io, s.inds)

function myempty(::Type{SimpleVectorIndexSetPattern{S}}) where {S}
    return SimpleVectorIndexSetPattern{S}(myempty(S))
end
function seed(::Type{SimpleVectorIndexSetPattern{S}}, i) where {S}
    return SimpleVectorIndexSetPattern{S}(seed(S, i))
end

# Tracer compatibility
inputs(s::SimpleVectorIndexSetPattern) = s.inds
gradient(s::SimpleVectorIndexSetPattern) = s.inds

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

Matrix sparsity pattern represented by an index set ``{(i,j)}`` of non-zero values.
Supports set-like datastructures `S`.

## Expected interface

The passed set type `S` has to implement:
* `SparseConnectivityTracer.myempty`
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

All of these methods are already implemented for `AbstractSet`s from Julia Base,
which are most commonly used in `SimpleMatrixIndexSetPattern`.
Refer to the individual documentation of each function for more information. 

## Fields
$(TYPEDFIELDS)
"""
struct SimpleMatrixIndexSetPattern{S} <: AbstractMatrixPattern
    "Set of index tuples represting non-zero entries ``(i, j)``."
    inds::S
end
Base.show(io::IO, s::SimpleMatrixIndexSetPattern) = Base.show(io, s.inds)

function myempty(::Type{SimpleMatrixIndexSetPattern{S}}) where {S}
    return SimpleMatrixIndexSetPattern{S}(myempty(S))
end

hessian(p::SimpleMatrixIndexSetPattern) = p.inds

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
    CombinedVectorAndMatrixPattern(vec::AbstractVectorPattern, mat::AbstractMatrixPattern)

Vector and matrix sparsity pattern constructed by combining two separate vector and matrix representations.
"""
struct CombinedVectorAndMatrixPattern{V<:AbstractVectorPattern,M<:AbstractMatrixPattern} <:
       AbstractVectorAndMatrixPattern
    vec::V
    mat::M
end

function Base.show(io::IO, s::CombinedVectorAndMatrixPattern)
    println(io, "Vector: ", s.vec)
    println(io, "Matrix: ", s.mat)
    return nothing
end

function myempty(::Type{CombinedVectorAndMatrixPattern{V,M}}) where {V,M}
    return CombinedVectorAndMatrixPattern{V,M}(myempty(V), myempty(M))
end
function seed(::Type{CombinedVectorAndMatrixPattern{V,M}}, index) where {V,M}
    return CombinedVectorAndMatrixPattern{V,M}(seed(V, index), myempty(M))
end

# Tracer compatibility
gradient(s::CombinedVectorAndMatrixPattern) = gradient(s.vec)
hessian(s::CombinedVectorAndMatrixPattern) = hessian(s.mat)
