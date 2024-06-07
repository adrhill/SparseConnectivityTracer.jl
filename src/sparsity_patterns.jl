"""
    AbstractSparsityPattern

Abstract supertype of all sparsity pattern representations.

## Expected interface

* `myempty(::Type{MyPattern})`: return an empty pattern
* `seed(::Type{MyPattern}, i::Integer)`: return an pattern that only contains the given index `i`
* `Base.show`

And for compatibility with tracers:

| Tracer               | Required accessor function                           |
|:---------------------|:-----------------------------------------------------|
| `ConnectivityTracer` | `inputs(p::MyPattern)`                               |
| `GradientTracer`     | `gradient(p::MyPattern)`                             |
| `HessianTracer`      | `gradient(p::MyPattern)` AND `hessian(p::MyPattern)` | 
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
"""
abstract type AbstractVectorPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

First order sparsity pattern represented by an index set of non-zero values.
Represented by a set of integer indices.

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

All of these methods are already implemented for `AbstractSet`s from Julia Base.
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

## Vector and Matrix

"""
    AbstractVectorAndMatrixPattern <: AbstractSparsityPattern

Abstract supertype of sparsity patterns representing matrices.
"""
abstract type AbstractVectorAndMatrixPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

Second order sparsity pattern represented by index sets of non-zero values.
Represented by a set of integer indices ``i`` and a set of tuples ``(i, j)``.

## Expected interface

The passed set type `F` has to implement:
* `SparseConnectivityTracer.myempty`
* `SparseConnectivityTracer.seed`
* `SparseConnectivityTracer.product`
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

The passed set type `S` has to implement:
* `SparseConnectivityTracer.myempty`
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

All of these methods are already implemented for `AbstractSet`s from Julia Base.
Refer to the individual documentation of each function for more information. 


## Fields
$(TYPEDFIELDS)
"""
struct SimpleVectorAndMatrixIndexSetPattern{F,S} <: AbstractVectorAndMatrixPattern
    "Set of indices represting non-zero entries ``i``."
    first_order::F
    "Set of index tuples represting non-zero entries ``(i, j)``."
    second_order::S
end
function Base.show(io::IO, s::SimpleVectorAndMatrixIndexSetPattern)
    println(io, "First  order: ", s.first_order)
    println(io, "Second order: ", s.second_order)
    return nothing
end

function myempty(::Type{SimpleVectorAndMatrixIndexSetPattern{F,S}}) where {F,S}
    return SimpleVectorAndMatrixIndexSetPattern{F,S}(myempty(F), myempty(S))
end
function seed(::Type{SimpleVectorAndMatrixIndexSetPattern{F,S}}, index) where {F,S}
    return SimpleVectorAndMatrixIndexSetPattern{F,S}(seed(F, index), myempty(S))
end

# Tracer compatibility
gradient(s::SimpleVectorAndMatrixIndexSetPattern) = s.first_order
hessian(s::SimpleVectorAndMatrixIndexSetPattern) = s.second_order
