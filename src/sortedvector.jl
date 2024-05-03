"""
    SortedVector

A wrapper for sorted vectors, designed for fast unions.

# Constructor

    SortedVector(data::AbstractVector; already_sorted=false)

# Example

```jldoctest
x = SortedVector([3, 4, 2])
x = SortedVector([1, 3, 5]; already_sorted=true)
z = union(x, y)

# output

SortedVector([1, 2, 3, 4, 5])
````
"""
struct SortedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    data::V

    function SortedVector{T,V}(data::V; already_sorted=false) where {T,V<:AbstractVector{T}}
        if already_sorted
            new{T,V}(data)
        else
            new{T,V}(sort(data))
        end
    end
end

Base.eltype(::SortedVector{T}) where {T} = T
Base.size(v::SortedVector) = size(v.data)
Base.getindex(v::SortedVector, i) = v.data[i]
Base.IndexStyle(::Type{SortedVector{T,V}}) where {T,V} = IndexStyle(V)

function SortedVector(data::V; already_sorted=false) where {T,V<:AbstractVector{T}}
    return SortedVector{T,V}(data; already_sorted)
end

Base.show(io::IO, sv::SortedVector) = print(io, "SortedVector($(sv.data))")

function Base.union(v1::SortedVector{T,V}, v2::SortedVector{T,V}) where {T,V}
    left, right = v1.data, v2.data
    both = similar(left, length(left) + length(right))
    left_index, right_index, both_index = 1, 1, 1
    # common part of left and right
    @inbounds while (
        left_index in eachindex(left) &&
        right_index in eachindex(right) &&
        both_index in eachindex(both)
    )
        left_item = left[left_index]
        right_item = right[right_index]
        left_smaller = left_item <= right_item
        right_smaller = right_item <= left_item
        both_item = ifelse(left_smaller, left_item, right_item)
        both[both_index] = both_item
        both_index += 1
        left_index = ifelse(left_smaller, left_index + 1, left_index)
        right_index = ifelse(right_smaller, right_index + 1, right_index)
    end
    # either left or right has reached its end at this point
    @inbounds while left_index in eachindex(left) && both_index in eachindex(both)
        both[both_index] = left[left_index]
        left_index += 1
        both_index += 1
    end
    @inbounds while right_index in eachindex(right) && both_index in eachindex(both)
        both[both_index] = right[right_index]
        right_index += 1
        both_index += 1
    end
    resize!(both, both_index - 1)
    return SortedVector(both; already_sorted=true)
end
