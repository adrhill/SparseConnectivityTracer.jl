"""
    DuplicateVector

Vector that can have duplicate values, for which union is just concatenation.
"""
struct DuplicateVector{T<:Number}
    data::Vector{T}

    DuplicateVector{T}(data::AbstractVector{T}) where {T} = new{T}(convert(Vector{T}, data))
    DuplicateVector{T}(x::Number) where {T} = new{T}([convert(T, x)])
    DuplicateVector{T}() where {T} = new{T}(T[])
end

Base.eltype(::Type{DuplicateVector{T}}) where {T} = T

Base.collect(dv::DuplicateVector) = collect(Set(dv.data))
