function keys2set(::Type{G}, d::H) where {G,H<:Dict}
    return G(keys(d))
end
