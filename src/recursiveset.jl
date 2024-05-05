"""
    RecursiveSet

A lazy union of sets.

# Constructors

    RecursiveSet(s::AbstractSet)
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
        return new{T}(nothing, nothing, nothing)
    end

    function RecursiveSet{T}(rs1::RecursiveSet{T}, rs2::RecursiveSet{T}) where {T}
        return new{T}(nothing, rs1, rs2)
    end
end

function Base.show(io::IO, rs::RecursiveSet{T}) where {T}
    if isnothing(rs.s) && isnothing(rs.child1) && isnothing(rs.child2)
        print(io, "RecursiveSet{$T} with no elements")
    elseif !isnothing(rs.s)
        print(io, "RecursiveSet{$T}($(rs.s))")
    else
        print(io, "RecursiveSet{$T} with two children")
    end
end

RecursiveSet(s) = RecursiveSet{eltype(s)}(s)
RecursiveSet(x::Number) = RecursiveSet{typeof(x)}(x)

Base.eltype(::Type{RecursiveSet{T}}) where {T} = T

function Base.union(rs1::RecursiveSet{T}, rs2::RecursiveSet{T}) where {T}
    return RecursiveSet{T}(rs1, rs2)
end

function Base.collect(rs::RecursiveSet{T}) where {T}
    if isnothing(rs.s) && isnothing(rs.child1) && isnothing(rs.child2)
        return T[]
    elseif !isnothing(rs.s)
        return collect(rs.s)
    else
        s_noduplicates = Set{T}()
        for l in AT.Leaves(rs)
            if !isnothing(l.s)
                union!(s_noduplicates, l.s)
            end
        end
        return collect(s_noduplicates)
    end
end

## Tree implementation

AT.childtype(::Type{RecursiveSet{T}}) where {T} = RecursiveSet{T}

function AT.children(rs::RecursiveSet{T}) where {T}
    if !isnothing(rs.child1) && !isnothing(rs.child2)
        return (rs.child1, rs.child2)
    else
        return Tuple{}()
    end
end
