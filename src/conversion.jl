## Type conversions
for T in (:JacobianTracer, :ConnectivityTracer, :HessianTracer)
    @eval Base.promote_rule(
        ::Type{$T{S}}, ::Type{N}
    ) where {N<:Number,S<:AbstractIndexSet} = $T{S}
    @eval Base.promote_rule(
        ::Type{N}, ::Type{$T{S}}
    ) where {N<:Number,S<:AbstractIndexSet} = $T{S}

    @eval Base.big(::Type{$T{S}}) where {S<:AbstractIndexSet}   = $T{S}
    @eval Base.widen(::Type{$T{S}}) where {S<:AbstractIndexSet} = $T{S}
    @eval Base.widen(t::$T)                                     = t

    @eval Base.convert(::Type{$T{S}}, x::Number) where {S<:AbstractIndexSet}   = empty($T{S})
    @eval Base.convert(::Type{$T{S}}, t::$T{S}) where {S<:AbstractIndexSet}    = t
    @eval Base.convert(::Type{<:Number}, t::$T{S}) where {S<:AbstractIndexSet} = t

    ## Constants
    @eval Base.zero(::Type{$T{S}}) where {S<:AbstractIndexSet}    = empty($T{S})
    @eval Base.one(::Type{$T{S}}) where {S<:AbstractIndexSet}     = empty($T{S})
    @eval Base.typemin(::Type{$T{S}}) where {S<:AbstractIndexSet} = empty($T{S})
    @eval Base.typemax(::Type{$T{S}}) where {S<:AbstractIndexSet} = empty($T{S})

    ## Array constructors
    @eval Base.similar(a::Array{$T{S},1}) where {S<:AbstractIndexSet}                       = zeros($T{S}, size(a, 1))
    @eval Base.similar(a::Array{$T{S},2}) where {S<:AbstractIndexSet}                       = zeros($T{S}, size(a, 1), size(a, 2))
    @eval Base.similar(a::Array{A,1}, ::Type{$T{S}}) where {A,S<:AbstractIndexSet}          = zeros($T{S}, size(a, 1))
    @eval Base.similar(a::Array{A,2}, ::Type{$T{S}}) where {A,S<:AbstractIndexSet}          = zeros($T{S}, size(a, 1), size(a, 2))
    @eval Base.similar(::Array{$T{S}}, m::Int) where {S<:AbstractIndexSet}                  = zeros($T{S}, m)
    @eval Base.similar(::Array, ::Type{$T{S}}, dims::Dims{N}) where {N,S<:AbstractIndexSet} = zeros($T{S}, dims)
    @eval Base.similar(::Array{$T{S}}, dims::Dims{N}) where {N,S<:AbstractIndexSet}         = zeros($T{S}, dims)
end
