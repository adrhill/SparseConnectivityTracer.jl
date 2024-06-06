abstract type AbstractSparsityPattern end
#=
Expected interface:
  * myempty(::Type{MyPattern}): return an empty pattern
  * seed(::Type{MyPattern}, i::Integer): return an pattern that only contains the given index `i`
  * Base.show

And for compatibility with...
  * ConnectivityTracer: inputs(p::MyPattern)   
  * GradientTracer:     gradient(p::MyPattern) 
  * HessianTracer:      gradient(p::MyPattern) AND hessian(p::MyPattern) 
=#

# Utilities on AbstractSet
myempty(::Type{S}) where {S<:AbstractSet} = S()
seed(::Type{S}, i::Integer) where {S<:AbstractSet} = S([i])

## First order
abstract type AbstractFirstOrderPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

First order sparsity pattern represented by an index set of non-zero values.
Represented by an `AbstractSet` of integer indices.

The passed set type `S` has to implement:
* `myempty(S)`: constructor for empty set (already implemented for `AbstractSet`)
* `seed(S, i)`: constructor for a set that only contains the given index `i`
* `product(a::S{T}, b::S{T})::S{Tuple{T,T}}`: inner product of sets
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

## Fields
$(TYPEDFIELDS)
"""
struct SetIndexset{S} <: AbstractFirstOrderPattern
    "Set of indices represting non-zero entries ``i``."
    inds::S
end
Base.show(io::IO, s::SetIndexset) = Base.show(io, s.inds)

myempty(::Type{SetIndexset{S}}) where {S} = SetIndexset{S}(myempty(S))
seed(::Type{SetIndexset{S}}, i) where {S} = SetIndexset{S}(seed(S, i))

# Tracer compatibility
inputs(s::SetIndexset) = s.inds
gradient(s::SetIndexset) = s.inds

## Second order
abstract type AbstractSecondOrderPattern <: AbstractSparsityPattern end

"""
$(TYPEDEF)

Second order sparsity pattern represented by index sets of non-zero values.
Represented by two `AbstractSet`s of integer indices ``i`` and tuples ``(i, j)``.

The passed set types `F` and `S` has to implement:
* `myempty(S)`: constructor for empty set (already implemented for `AbstractSet`)
* `seed(S, i)`: constructor for a set that only contains the given index `i`
* `product(a::S{T}, b::S{T})::S{Tuple{T,T}}`: inner product of sets
* `Base.union`
* `Base.union!`
* `Base.iterate`
* `Base.collect`
* `Base.show`

## Fields
$(TYPEDFIELDS)
"""
struct DualSetIndexset{F,S} <: AbstractSecondOrderPattern
    "Set of indices represting non-zero entries ``i``."
    first_order::F
    "Set of index tuples represting non-zero entries ``(i, j)``."
    second_order::S
end
function Base.show(io::IO, s::DualSetIndexset)
    println(io, "First  order: ", s.first_order)
    println(io, "Second order: ", s.second_order)
    return nothing
end

function myempty(::Type{DualSetIndexset{F,S}}) where {F,S}
    return DualSetIndexset{F,S}(myempty(F), myempty(S))
end
function seed(::Type{DualSetIndexset{F,S}}, index) where {F,S}
    return DualSetIndexset{F,S}(seed(F, index), myempty(S))
end

# Tracer compatibility
gradient(s::DualSetIndexset) = s.first_order
hessian(s::DualSetIndexset) = s.second_order
