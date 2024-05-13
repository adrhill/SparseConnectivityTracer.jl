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

function Base.union(dv1::DuplicateVector{T}, dv2::DuplicateVector{T}) where {T}
    return DuplicateVector{T}(vcat(dv1.data, dv2.data))
end

Base.iterate(dv::DuplicateVector)             = iterate(collect(dv))
Base.iterate(dv::DuplicateVector, i::Integer) = iterate(collect(dv), i)
