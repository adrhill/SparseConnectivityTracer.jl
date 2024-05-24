"""
    RecursiveSet

Lazy union of sets.
"""
struct RecursiveSet{T} <: AbstractSet{T}
    s::Union{Nothing,Set{T}}
    child1::Union{Nothing,RecursiveSet{T}}
    child2::Union{Nothing,RecursiveSet{T}}

    function RecursiveSet{T}(s::Union{AbstractSet,AbstractVector}) where {T}
        return new{T}(Set{T}(s), nothing, nothing)
    end

    function RecursiveSet{T}(x) where {T}
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

function Base.show(io::IO, rs::RecursiveSet)
    return print_recursiveset(io, rs; offset=0)
end

function Base.show(io::IO, ::MIME"text/plain", rs::RecursiveSet)
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

Base.iterate(rs::RecursiveSet)             = iterate(collect(rs))
Base.iterate(rs::RecursiveSet, i::Integer) = iterate(collect(rs), i)

function ×(
    ::Type{RecursiveSet{Tuple{T,T}}}, a::RecursiveSet{T}, b::RecursiveSet{T}
) where {T}
    # TODO: slow
    return RecursiveSet{Tuple{T,T}}(vec(collect(Iterators.product(collect(a), collect(b)))))
end

function ×(a::RecursiveSet{T}, b::RecursiveSet{T}) where {T}
    return ×(RecursiveSet{Tuple{T,T}}, a, b)
end
