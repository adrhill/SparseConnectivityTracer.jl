"""
    RecursiveSet

Lazy union of sets.
"""
struct RecursiveSet{T<:Number}
    s::Union{Nothing,Set{T}}
    child1::Union{Nothing,RecursiveSet{T}}
    child2::Union{Nothing,RecursiveSet{T}}

    function RecursiveSet{T}(s) where {T}
        return new{T}(Set{T}(s), nothing, nothing)
    end

    function RecursiveSet{T}(x::Number) where {T}
        return new{T}(Set{T}(convert(T, x)), nothing, nothing)
    end

    function RecursiveSet{T}() where {T}
        return new{T}(Set{T}(), nothing, nothing)
    end

    function RecursiveSet{T}(rs1::RecursiveSet{T}, rs2::RecursiveSet{T}) where {T}
        return new{T}(nothing, rs1, rs2)
    end
end

function print_recursiveset(io::IO, rs::RecursiveSet{T}; offset) where {T}
    if !isnothing(rs.s)
        print(io, "RecursiveSet{$T} containing $(rs.s)")
    else
        print(io, "RecursiveSet{$T} with two children:")
        print(io, "\n  ", " "^offset, "1: ")
        print_recursiveset(io, rs.child1; offset=offset + 2)
        print(io, "\n  ", " "^offset, "2: ")
        print_recursiveset(io, rs.child2; offset=offset + 2)
    end
end

function Base.show(io::IO, rs::RecursiveSet{T}) where {T}
    return print_recursiveset(io, rs; offset=0)
end

Base.eltype(::Type{RecursiveSet{T}}) where {T} = T

function Base.union(rs1::RecursiveSet{T}, rs2::RecursiveSet{T}) where {T}
    return RecursiveSet{T}(rs1, rs2)
end

function Base.collect(rs::RecursiveSet{T}) where {T}
    accumulator = Set{T}()
    collect_aux!(accumulator, rs)
    return collect(accumulator)
end

function collect_aux!(accumulator::Set{T}, rs::RecursiveSet{T})::Nothing where {T}
    if !isnothing(rs.s)
        union!(accumulator, rs.s::Set{T})
    else
        collect_aux!(accumulator, rs.child1::RecursiveSet{T})
        collect_aux!(accumulator, rs.child2::RecursiveSet{T})
    end
    return nothing
end

## SCT tricks

function keys2set(::Type{S}, d::Dict{I}) where {I<:Integer,S<:RecursiveSet{I}}
    return S(keys(d))
end

const EMPTY_CONNECTIVITY_TRACER_RS_U16 = ConnectivityTracer(RecursiveSet{UInt16}())
const EMPTY_CONNECTIVITY_TRACER_RS_U32 = ConnectivityTracer(RecursiveSet{UInt32}())
const EMPTY_CONNECTIVITY_TRACER_RS_U64 = ConnectivityTracer(RecursiveSet{UInt64}())

const EMPTY_JACOBIAN_TRACER_RS_U16 = JacobianTracer(RecursiveSet{UInt16}())
const EMPTY_JACOBIAN_TRACER_RS_U32 = JacobianTracer(RecursiveSet{UInt32}())
const EMPTY_JACOBIAN_TRACER_RS_U64 = JacobianTracer(RecursiveSet{UInt64}())

const EMPTY_HESSIAN_TRACER_RS_U16 = HessianTracer(Dict{UInt16,RecursiveSet{UInt16}}())
const EMPTY_HESSIAN_TRACER_RS_U32 = HessianTracer(Dict{UInt32,RecursiveSet{UInt32}}())
const EMPTY_HESSIAN_TRACER_RS_U64 = HessianTracer(Dict{UInt64,RecursiveSet{UInt64}}())

function empty(::Type{ConnectivityTracer{UInt16,RecursiveSet{UInt16}}})
    return EMPTY_CONNECTIVITY_TRACER_RS_U16
end
function empty(::Type{ConnectivityTracer{UInt32,RecursiveSet{UInt32}}})
    return EMPTY_CONNECTIVITY_TRACER_RS_U32
end
function empty(::Type{ConnectivityTracer{UInt64,RecursiveSet{UInt64}}})
    return EMPTY_CONNECTIVITY_TRACER_RS_U64
end

empty(::Type{JacobianTracer{UInt16,RecursiveSet{UInt16}}}) = EMPTY_JACOBIAN_TRACER_RS_U16
empty(::Type{JacobianTracer{UInt32,RecursiveSet{UInt32}}}) = EMPTY_JACOBIAN_TRACER_RS_U32
empty(::Type{JacobianTracer{UInt64,RecursiveSet{UInt64}}}) = EMPTY_JACOBIAN_TRACER_RS_U64

function empty(
    ::Type{HessianTracer{UInt16,RecursiveSet{UInt16},Dict{UInt16,RecursiveSet{UInt16}}}}
)
    return EMPTY_HESSIAN_TRACER_RS_U16
end
function empty(
    ::Type{HessianTracer{UInt32,RecursiveSet{UInt32},Dict{UInt32,RecursiveSet{UInt32}}}}
)
    return EMPTY_HESSIAN_TRACER_RS_U32
end
function empty(
    ::Type{HessianTracer{UInt64,RecursiveSet{UInt64},Dict{UInt64,RecursiveSet{UInt64}}}}
)
    return EMPTY_HESSIAN_TRACER_RS_U64
end
