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
function keys2set(::Type{RecursiveSet{T}}, d::Dict{T,S}) where {T,S}
    return RecursiveSet{T}(collect(keys(d)))
end

function keys2set(::Type{RecursiveSet{T}}, d::Dict{I,S}) where {T,I,S}
    return RecursiveSet{T}(convert.(T, keys(d)))
end
