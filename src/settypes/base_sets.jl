function keys2set(::Type{S}, d::Dict{I}) where {I<:Integer,S<:AbstractSet{<:I}}
    return S(keys(d))
end
