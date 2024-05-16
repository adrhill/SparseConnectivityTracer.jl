## Type conversions (non-dual)
for TT in (:GradientTracer, :ConnectivityTracer, :HessianTracer)
    @eval Base.promote_rule(::Type{T}, ::Type{N}) where {T<:$TT,N<:Number} = T
    @eval Base.promote_rule(::Type{N}, ::Type{T}) where {T<:$TT,N<:Number} = T

    @eval Base.big(::Type{T}) where {T<:$TT}   = T
    @eval Base.big(t::T) where {T<:$TT}        = t
    @eval Base.widen(::Type{T}) where {T<:$TT} = T
    @eval Base.widen(t::T) where {T<:$TT}      = t

    @eval Base.convert(::Type{T}, x::Number) where {T<:$TT}   = empty(T)
    @eval Base.convert(::Type{T}, t::T) where {T<:$TT}        = t
    @eval Base.convert(::Type{<:Number}, t::T) where {T<:$TT} = t

    ## Constants
    @eval Base.zero(::Type{T}) where {T<:$TT}        = empty(T)
    @eval Base.zero(::T) where {T<:$TT}              = empty(T)
    @eval Base.one(::Type{T}) where {T<:$TT}         = empty(T)
    @eval Base.one(::T) where {T<:$TT}               = empty(T)
    @eval Base.typemin(::Type{T}) where {T<:$TT}     = empty(T)
    @eval Base.typemin(::T) where {T<:$TT}           = empty(T)
    @eval Base.typemax(::Type{T}) where {T<:$TT}     = empty(T)
    @eval Base.typemax(::T) where {T<:$TT}           = empty(T)
    @eval Base.eps(::Type{T}) where {T<:$TT}         = empty(T)
    @eval Base.eps(::T) where {T<:$TT}               = empty(T)
    @eval Base.floatmin(::Type{T}) where {T<:$TT}    = empty(T)
    @eval Base.floatmin(::T) where {T<:$TT}          = empty(T)
    @eval Base.floatmax(::Type{T}) where {T<:$TT}    = empty(T)
    @eval Base.floatmax(::T) where {T<:$TT}          = empty(T)
    @eval Base.maxintfloat(::Type{T}) where {T<:$TT} = empty(T)
    @eval Base.maxintfloat(::T) where {T<:$TT}       = empty(T)

    ## Array constructors
    @eval Base.similar(a::Array{T,1}) where {T<:$TT}               = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{T,2}) where {T<:$TT}               = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(a::Array{A,1}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{A,2}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(::Array{T}, m::Int) where {T<:$TT}          = zeros(T, m)
    @eval Base.similar(::Array{T}, dims::Dims{N}) where {N,T<:$TT} = zeros(T, dims)
end

function Base.similar(::Array, ::Type{ConnectivityTracer{C}}, dims::Dims{N}) where {C,N}
    return zeros(ConnectivityTracer{C}, dims)
end
function Base.similar(::Array, ::Type{GradientTracer{G}}, dims::Dims{N}) where {G,N}
    return zeros(GradientTracer{G}, dims)
end
function Base.similar(::Array, ::Type{HessianTracer{G,H}}, dims::Dims{N}) where {G,H,N}
    return zeros(HessianTracer{G,H}, dims)
end
