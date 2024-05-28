"""
    DuplicateVector

Vector that can have duplicate values, for which union is just concatenation.
"""
struct DuplicateVector{T} <: AbstractSet{T}
    data::Vector{T}

    DuplicateVector{T}(data::AbstractVector) where {T} = new{T}(convert(Vector{T}, data))
    DuplicateVector{T}(x) where {T} = new{T}([convert(T, x)])
    DuplicateVector{T}() where {T} = new{T}(T[])
end

function Base.show(io::IO, dv::DuplicateVector)
    return print(io, "DuplicateVector($(dv.data))")
end

function Base.show(io::IO, ::MIME"text/plain", dv::DuplicateVector)
    return print(io, "DuplicateVector($(dv.data))")
end

Base.eltype(::Type{DuplicateVector{T}}) where {T} = T

Base.collect(dv::DuplicateVector) = unique!(dv.data)

function Base.union!(a::S, b::S) where {S<:DuplicateVector}
    append!(a.data, b.data)
    return a
end

function Base.union(a::S, b::S) where {S<:DuplicateVector}
    return S(vcat(a.data, b.data))
end

Base.iterate(dv::DuplicateVector)             = iterate(collect(dv))
Base.iterate(dv::DuplicateVector, i::Integer) = iterate(collect(dv), i)

function ×(a::DuplicateVector{T}, b::DuplicateVector{T}) where {T}
    return DuplicateVector{Tuple{T,T}}(vec(collect(Iterators.product(a.data, b.data))))
end
