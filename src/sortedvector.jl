"""
    SortedVector

A wrapper for sorted vectors, designed for fast unions.

# Constructor

    SortedVector(data::AbstractVector; already_sorted=true)

# Example

```jldoctest
x = SortedVector([1, 3, 5])
y = SortedVector([3, 4, 2]; already_sorted=false)
z = union(x, y)

# output

hello
````
"""
struct SortedVector{T,V<:AbstractVector{T}}
    data::V
    function SortedVector(data::V; already_sorted=true) where {T,V<:AbstractVector{T}}
        if already_sorted
            new{T,V}(data)
        else
            new{T,V}(sort(data))
        end
    end
end

Base.show(io::IO, sv::SortedVector) = print(io, "SortedVector($(sv.data))")

function Base.union(v1::SortedVector{T}, v2::SortedVector{T}) where {T}
    x, y = v1.data, v2.data
    z = similar(x, length(x) + length(y))
    return SortedVector(z)
end
