"""
    DuplicateVector

Vector that can have duplicate values, for which union is just concatenation.
"""
struct DuplicateVector{T<:Number} <: AbstractSet{T}
    data::Vector{T}

    DuplicateVector{T}(data::AbstractVector) where {T} = new{T}(convert(Vector{T}, data))
    DuplicateVector{T}(x::Number) where {T} = new{T}([convert(T, x)])
    DuplicateVector{T}() where {T} = new{T}(T[])
end

Base.eltype(::Type{DuplicateVector{T}}) where {T} = T

Base.collect(dv::DuplicateVector) = unique!(dv.data)

function Base.union(a::S, b::S) where {S<:DuplicateVector}
    return S(vcat(a.data, b.data))
end

function Base.union!(a::S, b::S) where {S<:DuplicateVector}
    return append!(a.data, b.data)
end

Base.iterate(dv::DuplicateVector)             = iterate(collect(dv))
Base.iterate(dv::DuplicateVector, i::Integer) = iterate(collect(dv), i)

# TODO: required by `Base.Iterators.ProductIterator` called in method `×` in src/tracers.jl.
# This is very slow and should be replaced by a custom `×` on `DuplicateVector`s.
Base.length(dv::DuplicateVector) = length(unique!(dv.data))
