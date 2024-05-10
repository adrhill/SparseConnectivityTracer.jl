## Type conversions
for TT in (:GradientTracer, :ConnectivityTracer, :HessianTracer)
    @eval Base.promote_rule(::Type{T}, ::Type{N}) where {T<:$TT,N<:Number} = T
    @eval Base.promote_rule(::Type{N}, ::Type{T}) where {T<:$TT,N<:Number} = T

    @eval Base.big(::Type{T}) where {T<:$TT}   = T
    @eval Base.widen(::Type{T}) where {T<:$TT} = T
    @eval Base.widen(t::T) where {T<:$TT}      = t

    @eval Base.convert(::Type{T}, x::Number) where {T<:$TT}   = empty(T)
    @eval Base.convert(::Type{T}, t::T) where {T<:$TT}        = t
    @eval Base.convert(::Type{<:Number}, t::T) where {T<:$TT} = t

    ## Constants
    @eval Base.zero(::Type{T}) where {T<:$TT}    = empty(T)
    @eval Base.one(::Type{T}) where {T<:$TT}     = empty(T)
    @eval Base.typemin(::Type{T}) where {T<:$TT} = empty(T)
    @eval Base.typemax(::Type{T}) where {T<:$TT} = empty(T)

    ## Array constructors
    @eval Base.similar(a::Array{T,1}) where {T<:$TT}               = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{T,2}) where {T<:$TT}               = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(a::Array{A,1}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{A,2}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(::Array{T}, m::Int) where {T<:$TT}          = zeros(T, m)
    @eval Base.similar(::Array{T}, dims::Dims{N}) where {N,T<:$TT} = zeros(T, dims)
end

function Base.similar(::Array, ::Type{ConnectivityTracer{I,S}}, dims::Dims{N}) where {I,S,N}
    return zeros(ConnectivityTracer{I,S}, dims)
end
function Base.similar(::Array, ::Type{GradientTracer{I,S}}, dims::Dims{N}) where {I,S,N}
    return zeros(GradientTracer{I,S}, dims)
end
function Base.similar(::Array, ::Type{HessianTracer{I,S,D}}, dims::Dims{N}) where {I,S,D,N}
    return zeros(HessianTracer{I,S,D}, dims)
end
