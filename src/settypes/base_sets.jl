function keys2set(::Type{G}, d::Dict) where {G}
    T = eltype(G)
    return G(convert.(T, keys(d)))
end
