"""
    SortedVector

Sorted vector without duplicates, designed for fast set unions with merging.
"""
struct SortedVector{T} <: AbstractSet{T}
    data::Vector{T}

    function SortedVector{T}(data::AbstractVector; sorted=false) where {T}
        sorted_data = if sorted
            data
        else
            sort(data)
        end
        return new{T}(convert(Vector{T}, sorted_data))
    end

    function SortedVector{T}(x) where {T}
        return new{T}([convert(T, x)])
    end

    function SortedVector{T}() where {T}
        return new{T}(T[])
    end
end

function Base.convert(::Type{SortedVector{T}}, v::Vector{T}) where {T}
    return SortedVector{T}(v; sorted=false)
end

Base.show(io::IO, v::SortedVector) = print(io, "SortedVector($(v.data))")

function Base.show(io::IO, ::MIME"text/plain", dv::SortedVector)
    return print(io, "SortedVector($(dv.data))")
end

Base.eltype(::Type{SortedVector{T}}) where {T} = T
Base.length(v::SortedVector) = length(v.data)
Base.copy(v::SortedVector{T}) where {T} = SortedVector{T}(copy(v.data); sorted=true)

function merge_sorted!(result::Vector{T}, left::Vector{T}, right::Vector{T}) where {T}
    resize!(result, length(left) + length(right))
    left_index, right_index, result_index = 1, 1, 1
    # common part of left and right
    while (left_index in eachindex(left) && right_index in eachindex(right))
        left_item = left[left_index]
        right_item = right[right_index]
        left_smaller = left_item <= right_item
        right_smaller = right_item <= left_item
        result_item = ifelse(left_smaller, left_item, right_item)
        result[result_index] = result_item
        left_index = ifelse(left_smaller, left_index + 1, left_index)
        right_index = ifelse(right_smaller, right_index + 1, right_index)
        result_index += 1
    end
    # either left or right has reached its end at this point
    while left_index in eachindex(left)
        result[result_index] = left[left_index]
        left_index += 1
        result_index += 1
    end
    while right_index in eachindex(right)
        result[result_index] = right[right_index]
        right_index += 1
        result_index += 1
    end
    resize!(result, result_index - 1)
    return result
end

function merge_sorted!(result::Vector{T}, other::Vector{T}) where {T}
    return merge_sorted!(result, copy(result), other)
end

function merge_sorted(left::Vector{T}, right::Vector{T}) where {T}
    result = similar(left, length(left) + length(right))
    merge_sorted!(result, left, right)
    return result
end

function Base.union(v1::SortedVector{T}, v2::SortedVector{T}) where {T}
    return SortedVector{T}(merge_sorted(v1.data, v2.data); sorted=true)
end

function Base.union!(v1::SortedVector{T}, v2::SortedVector{T}) where {T}
    merge_sorted!(v1.data, v2.data)
    return v1
end

Base.collect(v::SortedVector) = v.data

Base.iterate(v::SortedVector) = iterate(v.data)
Base.iterate(v::SortedVector, i::Integer) = iterate(v.data, i)

function product(v1::SortedVector{T}, v2::SortedVector{T}) where {T}
    prod_data = vec(collect((i, j) for i in v1.data, j in v2.data if i <= j))
    return SortedVector{Tuple{T,T}}(prod_data; sorted=true)
end
