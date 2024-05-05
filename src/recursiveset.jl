"""
    RecursiveSet

A lazy union of sets.

# Constructors

    RecursiveSet(x::Number)
    RecursiveSet(s1::RecursiveSet, s2::RecursiveSet)
"""
struct RecursiveSet{T<:Number}
    x::Union{Nothing,T}
    s1::Union{Nothing,RecursiveSet{T}}
    s2::Union{Nothing,RecursiveSet{T}}

    function RecursiveSet{T}(s1::RecursiveSet{T}, s2::RecursiveSet{T}) where {T}
        return new{T}(nothing, s1, s2)
    end

    function RecursiveSet{T}(x::Number) where {T}
        return new{T}(convert(T, x), nothing, nothing)
    end

    function RecursiveSet{T}() where {T}
        return new{T}(nothing, nothing, nothing)
    end
end

function Base.show(io::IO, s::RecursiveSet{T}) where {T}
    if !isnothing(s.x)
        print(io, "RecursiveSet{$T}($(s.x))")
    else
        print(io, "RecursiveSet{$T} with two children")
    end
end

function RecursiveSet(s1::RecursiveSet{T}, s2::RecursiveSet{T}) where {T}
    return RecursiveSet{T}(s1, s2)
end

function RecursiveSet(x::Number)
    return RecursiveSet{typeof(x)}(x)
end

Base.eltype(::RecursiveSet{T}) where {T} = T

function Base.union(s1::RecursiveSet{T}, s2::RecursiveSet{T}) where {T}
    return RecursiveSet{T}(s1, s2)
end

function Base.collect(s::RecursiveSet{T}) where {T}
    if isnothing(s.x) && isnothing(s.s1) && isnothing(s.s2)
        return T[]
    elseif !isnothing(s.x)
        return T[s.x]
    else
        s_noduplicates = Set{T}()
        for l in AT.Leaves(s)
            x = AT.nodevalue(l)
            if !isnothing(x)
                push!(s_noduplicates, x)
            end
        end
        return collect(s_noduplicates)
    end
end

## Tree implementation

AT.nodevalue(s::RecursiveSet) = s.x

AT.childtype(::Type{RecursiveSet{T}}) where {T} = RecursiveSet{T}

function AT.children(s::RecursiveSet{T}) where {T}
    if !isnothing(s.s1) && !isnothing(s.s2)
        return (s.s1, s.s2)
    else
        return Tuple{}()
    end
end
