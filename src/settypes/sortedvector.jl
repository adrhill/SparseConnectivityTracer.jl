"""
    SortedVector

Sorted vector without duplicates, designed for fast set unions with merging.
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

## SCT tricks

function keys2set(::Type{S}, d::Dict{I}) where {I<:Integer,S<:SortedVector{I}}
    return S(collect(keys(d)); sorted=false)
end

const EMPTY_CONNECTIVITY_TRACER_SV_U16  = ConnectivityTracer(SortedVector{UInt16}())
const EMPTY_CONNECTIVITY_TRACER_SV_U32  = ConnectivityTracer(SortedVector{UInt32}())
const EMPTY_CONNECTIVITY_TRACER_SV_U64  = ConnectivityTracer(SortedVector{UInt64}())

const EMPTY_JACOBIAN_TRACER_SV_U16  = JacobianTracer(SortedVector{UInt16}())
const EMPTY_JACOBIAN_TRACER_SV_U32  = JacobianTracer(SortedVector{UInt32}())
const EMPTY_JACOBIAN_TRACER_SV_U64  = JacobianTracer(SortedVector{UInt64}())

const EMPTY_HESSIAN_TRACER_SV_U16  = HessianTracer(Dict{UInt16,SortedVector{UInt16}}())
const EMPTY_HESSIAN_TRACER_SV_U32  = HessianTracer(Dict{UInt32,SortedVector{UInt32}}())
const EMPTY_HESSIAN_TRACER_SV_U64  = HessianTracer(Dict{UInt64,SortedVector{UInt64}}())

empty(::Type{ConnectivityTracer{SortedVector{UInt16}}}) = EMPTY_CONNECTIVITY_TRACER_SV_U16
empty(::Type{ConnectivityTracer{SortedVector{UInt32}}}) = EMPTY_CONNECTIVITY_TRACER_SV_U32
empty(::Type{ConnectivityTracer{SortedVector{UInt64}}}) = EMPTY_CONNECTIVITY_TRACER_SV_U64

empty(::Type{JacobianTracer{SortedVector{UInt16}}}) = EMPTY_JACOBIAN_TRACER_SV_U16
empty(::Type{JacobianTracer{SortedVector{UInt32}}}) = EMPTY_JACOBIAN_TRACER_SV_U32
empty(::Type{JacobianTracer{SortedVector{UInt64}}}) = EMPTY_JACOBIAN_TRACER_SV_U64

empty(::Type{HessianTracer{SortedVector{UInt16},UInt16}}) = EMPTY_HESSIAN_TRACER_SV_U16
empty(::Type{HessianTracer{SortedVector{UInt32},UInt32}}) = EMPTY_HESSIAN_TRACER_SV_U32
empty(::Type{HessianTracer{SortedVector{UInt64},UInt64}}) = EMPTY_HESSIAN_TRACER_SV_U64
