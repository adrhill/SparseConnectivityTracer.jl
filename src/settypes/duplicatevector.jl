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

const EMPTY_CONNECTIVITY_TRACER_DV_U16 = ConnectivityTracer(DuplicateVector{UInt16}())
const EMPTY_CONNECTIVITY_TRACER_DV_U32 = ConnectivityTracer(DuplicateVector{UInt32}())
const EMPTY_CONNECTIVITY_TRACER_DV_U64 = ConnectivityTracer(DuplicateVector{UInt64}())

const EMPTY_JACOBIAN_TRACER_DV_U16 = JacobianTracer(DuplicateVector{UInt16}())
const EMPTY_JACOBIAN_TRACER_DV_U32 = JacobianTracer(DuplicateVector{UInt32}())
const EMPTY_JACOBIAN_TRACER_DV_U64 = JacobianTracer(DuplicateVector{UInt64}())

const EMPTY_HESSIAN_TRACER_DV_U16 = HessianTracer(Dict{UInt16,DuplicateVector{UInt16}}())
const EMPTY_HESSIAN_TRACER_DV_U32 = HessianTracer(Dict{UInt32,DuplicateVector{UInt32}}())
const EMPTY_HESSIAN_TRACER_DV_U64 = HessianTracer(Dict{UInt64,DuplicateVector{UInt64}}())

function empty(::Type{ConnectivityTracer{DuplicateVector{UInt16}}})
    return EMPTY_CONNECTIVITY_TRACER_DV_U16
end
function empty(::Type{ConnectivityTracer{DuplicateVector{UInt32}}})
    return EMPTY_CONNECTIVITY_TRACER_DV_U32
end
function empty(::Type{ConnectivityTracer{DuplicateVector{UInt64}}})
    return EMPTY_CONNECTIVITY_TRACER_DV_U64
end

empty(::Type{JacobianTracer{DuplicateVector{UInt16}}}) = EMPTY_JACOBIAN_TRACER_DV_U16
empty(::Type{JacobianTracer{DuplicateVector{UInt32}}}) = EMPTY_JACOBIAN_TRACER_DV_U32
empty(::Type{JacobianTracer{DuplicateVector{UInt64}}}) = EMPTY_JACOBIAN_TRACER_DV_U64

empty(::Type{HessianTracer{DuplicateVector{UInt16},UInt16}}) = EMPTY_HESSIAN_TRACER_DV_U16
empty(::Type{HessianTracer{DuplicateVector{UInt32},UInt32}}) = EMPTY_HESSIAN_TRACER_DV_U32
empty(::Type{HessianTracer{DuplicateVector{UInt64},UInt64}}) = EMPTY_HESSIAN_TRACER_DV_U64
