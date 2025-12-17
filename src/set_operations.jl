#= Create empty sets and compute unions of sets and cross-products of sets =#

myempty(::S) where {S <: AbstractSet} = S()
myempty(::Type{S}) where {S <: AbstractSet} = S()
myempty(::D) where {D <: AbstractDict} = D()
myempty(::Type{D}) where {D <: AbstractDict} = D()
seed(::Type{S}, i::Integer) where {S <: AbstractSet} = S(i)

myunion!(a::S, b::S) where {S <: AbstractSet} = union!(a, b)
function myunion!(a::D, b::D) where {I <: Integer, S <: AbstractSet{I}, D <: AbstractDict{I, S}}
    for k in keys(b)
        if haskey(a, k)
            union!(a[k], b[k])
        else
            push!(a, k => b[k])
        end
    end
    return a
end

# convert to set of index tuples
tuple_set(s::AbstractSet{Tuple{I, I}}) where {I <: Integer} = s
function tuple_set(d::AbstractDict{I, S}) where {I <: Integer, S <: AbstractSet{I}}
    return Set((k, v) for k in keys(d) for v in d[k])
end

""""
    product(a::S{T}, b::S{T})::S{Tuple{T,T}}

Inner product of set-like inputs `a` and `b`.
"""
function product(a::AbstractSet{I}, b::AbstractSet{I}) where {I <: Integer}
    # Since the Hessian is symmetric, we only have to keep track of index-tuples (i,j) with i≤j.
    return Set((i, j) for i in a, j in b if i <= j)
end

function union_product!(
        hessian::H, gradient_x::G, gradient_y::G
    ) where {I <: Integer, G <: AbstractSet{I}, H <: AbstractSet{Tuple{I, I}}}
    for i in gradient_x
        for j in gradient_y
            if i <= j # symmetric Hessian
                push!(hessian, (i, j))
            end
        end
    end
    return hessian
end

function union_product!(
        hessian::AbstractDict{I, S}, gradient_x::S, gradient_y::S
    ) where {I <: Integer, S <: AbstractSet{I}}
    for i in gradient_x
        if !haskey(hessian, i)
            push!(hessian, i => S())
        end
        for j in gradient_y
            if i <= j # symmetric Hessian
                push!(hessian[i], j)
            end
        end
    end
    return hessian
end
