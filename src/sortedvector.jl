"""
    SortedVector

A wrapper for sorted vectors, designed for fast unions.

# Constructor

    SortedVector(data::AbstractVector; sorted=false)

# Example

```jldoctest
x = SortedVector([3, 4, 2])
x = SortedVector([1, 3, 5]; sorted=true)
z = union(x, y)

# output

SortedVector([1, 2, 3, 4, 5])
````
"""
struct SortedVector{T<:Number} <: AbstractVector{T}
    data::Vector{T}

    function SortedVector{T}(data::AbstractVector{T}; sorted=false) where {T}
        sorted_data = ifelse(sorted, data, sort(data))
        return new{T}(convert(Vector{T}, sorted_data))
    end

    function SortedVector{T}(x::Number) where {T}
        return new{T}([convert(T, x)])
    end

    function SortedVector{T}() where {T}
        return new{T}(T[])
    end
end

function Base.convert(::Type{SortedVector{T}}, v::Vector{T}) where {T}
    return SortedVector{T}(v; sorted=false)
end

Base.eltype(::SortedVector{T}) where {T} = T
Base.size(v::SortedVector) = size(v.data)
Base.getindex(v::SortedVector, i) = v.data[i]
Base.IndexStyle(::Type{SortedVector{T}}) where {T} = IndexStyle(Vector{T})
Base.show(io::IO, v::SortedVector) = print(io, "SortedVector($(v.data))")

function Base.union(v1::SortedVector{T}, v2::SortedVector{T}) where {T}
    left, right = v1.data, v2.data
    result = similar(left, length(left) + length(right))
    left_index, right_index, result_index = 1, 1, 1
    # common part of left and right
    @inbounds while (
        left_index in eachindex(left) &&
        right_index in eachindex(right) &&
        result_index in eachindex(result)
    )
        left_item = left[left_index]
        right_item = right[right_index]
        left_smaller = left_item <= right_item
        right_smaller = right_item <= left_item
        result_item = ifelse(left_smaller, left_item, right_item)
        result[result_index] = result_item
        result_index += 1
        left_index = ifelse(left_smaller, left_index + 1, left_index)
        right_index = ifelse(right_smaller, right_index + 1, right_index)
    end
    # either left or right has reached its end at this point
    @inbounds while left_index in eachindex(left) && result_index in eachindex(result)
        result[result_index] = left[left_index]
        left_index += 1
        result_index += 1
    end
    @inbounds while right_index in eachindex(right) && result_index in eachindex(result)
        result[result_index] = right[right_index]
        right_index += 1
        result_index += 1
    end
    resize!(result, result_index - 1)
    return SortedVector{T}(result; sorted=true)
end
