"""
    RecursiveSet

A lazy union of sets.

# Constructors

    RecursiveSet(s::AbstractSet)
"""
mutable struct RecursiveSet{T<:Number}
    s::Union{Nothing,Set{T}}
    child1::RecursiveSet{T}
    child2::RecursiveSet{T}

    function RecursiveSet{T}(s) where {T}
        rs = new{T}(Set{T}(s))
        rs.child1 = rs
        rs.child2 = rs
        return rs
    end

    function RecursiveSet{T}(x::Number) where {T}
        rs = new{T}(Set{T}(convert(T, x)))
        rs.child1 = rs
        rs.child2 = rs
        return rs
    end

    function RecursiveSet{T}() where {T}
        rs = new{T}(Set{T}())
        rs.child1 = rs
        rs.child2 = rs
        return rs
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

RecursiveSet(s) = RecursiveSet{eltype(s)}(s)
RecursiveSet(x::Number) = RecursiveSet{typeof(x)}(x)

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
        collect_aux!(accumulator, rs.child1)
        collect_aux!(accumulator, rs.child2)
    end
    return nothing
end
