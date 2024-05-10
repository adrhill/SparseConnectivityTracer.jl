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

function Base.union(dv1::DuplicateVector{T}, dv2::DuplicateVector{T}) where {T}
    return DuplicateVector{T}(vcat(dv1.data, dv2.data))
end

Base.collect(dv::DuplicateVector) = collect(Set(dv.data))

## SCT tricks

function keys2set(::Type{S}, d::Dict{I}) where {I<:Integer,S<:DuplicateVector{I}}
    return S(collect(keys(d)))
end
