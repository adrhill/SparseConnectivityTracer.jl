#! format: off

## Type conversions (non-dual)
for TT in (GradientTracer, ConnectivityTracer, HessianTracer)
    Base.promote_rule(::Type{T}, ::Type{N}) where {T<:TT,N<:Number} = T
    Base.promote_rule(::Type{N}, ::Type{T}) where {T<:TT,N<:Number} = T

    Base.big(::Type{T})   where {T<:TT} = T
    Base.widen(::Type{T}) where {T<:TT} = T
    Base.big(t::T)        where {T<:TT} = t
    Base.widen(t::T)      where {T<:TT} = t

    Base.convert(::Type{T}, x::Number)   where {T<:TT} = empty(T)
    Base.convert(::Type{T}, t::T)        where {T<:TT} = t
    Base.convert(::Type{<:Number}, t::T) where {T<:TT} = t

    ## Constants
    Base.zero(::Type{T})        where {T<:TT} = empty(T)
    Base.one(::Type{T})         where {T<:TT} = empty(T)
    Base.typemin(::Type{T})     where {T<:TT} = empty(T)
    Base.typemax(::Type{T})     where {T<:TT} = empty(T)
    Base.eps(::Type{T})         where {T<:TT} = empty(T)
    Base.floatmin(::Type{T})    where {T<:TT} = empty(T)
    Base.floatmax(::Type{T})    where {T<:TT} = empty(T)
    Base.maxintfloat(::Type{T}) where {T<:TT} = empty(T)
    
    Base.zero(::T)        where {T<:TT} = empty(T)
    Base.one(::T)         where {T<:TT} = empty(T)
    Base.typemin(::T)     where {T<:TT} = empty(T)
    Base.typemax(::T)     where {T<:TT} = empty(T)
    Base.eps(::T)         where {T<:TT} = empty(T)
    Base.floatmin(::T)    where {T<:TT} = empty(T)
    Base.floatmax(::T)    where {T<:TT} = empty(T)
    Base.maxintfloat(::T) where {T<:TT} = empty(T)

    ## Array constructors
    Base.similar(a::Array{T,1})                     where {T<:TT}   = zeros(T, size(a, 1))
    Base.similar(a::Array{T,2})                     where {T<:TT}   = zeros(T, size(a, 1), size(a, 2))
    Base.similar(a::Array{A,1}, ::Type{T})          where {T<:TT,A} = zeros(T, size(a, 1))
    Base.similar(a::Array{A,2}, ::Type{T})          where {T<:TT,A} = zeros(T, size(a, 1), size(a, 2))
    Base.similar(::Array{T}, m::Int)                where {T<:TT}   = zeros(T, m)
    Base.similar(::Array{T}, dims::Dims{N})         where {T<:TT,N} = zeros(T, dims)
    Base.similar(::Array, ::Type{T}, dims::Dims{N}) where {T<:TT,N} = zeros(T, dims)
end

## Duals
function Base.promote_rule(::Type{D}, ::Type{N}) where {P,T,D<:Dual{P,T},N<:Number}
    PP = Base.promote_rule(P, N) # TODO: possible method call error?
    return D{PP,T}
end
function Base.promote_rule(::Type{N}, ::Type{D}) where {P,T,D<:Dual{P,T},N<:Number}
    PP = Base.promote_rule(P, N) # TODO: possible method call error?
    return D{PP,T}
end

Base.big(::Type{D})   where {P,T,D<:Dual{P,T}} = Dual{big(P),T}
Base.widen(::Type{D}) where {P,T,D<:Dual{P,T}} = Dual{widen(P),T}
Base.big(d::D)        where {P,T,D<:Dual{P,T}} = Dual(big(primal(d)),   tracer(d))
Base.widen(d::D)      where {P,T,D<:Dual{P,T}} = Dual(widen(primal(d)), tracer(d))

Base.convert(::Type{D}, x::Number) where {P,T,D<:Dual{P,T}}  = Dual(x, empty(T))
Base.convert(::Type{D}, d::D)      where {D<:Dual}           = d
Base.convert(::Type{T}, d::D)      where {T<:Number,D<:Dual} = Dual(convert(T, primal(d)), tracer(d))

## Constants
Base.zero(::Type{D})        where {P,T,D<:Dual{P,T}} = D(zero(P),        empty(T))
Base.one(::Type{D})         where {P,T,D<:Dual{P,T}} = D(one(P),         empty(T))
Base.typemin(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemin(P),     empty(T))
Base.typemax(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemax(P),     empty(T))
Base.eps(::Type{D})         where {P,T,D<:Dual{P,T}} = D(eps(P),         empty(T))
Base.floatmin(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmin(P),    empty(T))
Base.floatmax(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmax(P),    empty(T))
Base.maxintfloat(::Type{D}) where {P,T,D<:Dual{P,T}} = D(maxintfloat(P), empty(T))

Base.zero(d::D)        where {P,T,D<:Dual{P,T}} = D(zero(primal(d)),        empty(T))
Base.one(d::D)         where {P,T,D<:Dual{P,T}} = D(one(primal(d)),         empty(T))
Base.typemin(d::D)     where {P,T,D<:Dual{P,T}} = D(typemin(primal(d)),     empty(T))
Base.typemax(d::D)     where {P,T,D<:Dual{P,T}} = D(typemax(primal(d)),     empty(T))
Base.eps(d::D)         where {P,T,D<:Dual{P,T}} = D(eps(primal(d)),         empty(T))
Base.floatmin(d::D)    where {P,T,D<:Dual{P,T}} = D(floatmin(primal(d)),    empty(T))
Base.floatmax(d::D)    where {P,T,D<:Dual{P,T}} = D(floatmax(primal(d)),    empty(T))
Base.maxintfloat(d::D) where {P,T,D<:Dual{P,T}} = D(maxintfloat(primal(d)), empty(T))

## Array constructors
function Base.similar(a::Array{D,1}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D,2}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{A,1}, ::Type{D}) where {A,P,T,D<:Dual{P,T}}
    p_out = similar(a, P)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{A,2}, ::Type{D}) where {A,P,T,D<:Dual{P,T}}
    p_out = similar(a, P)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D}, m::Int) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a), m)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D}, dims::Dims{N}) where {N,D<:Dual}
    p_out = similar(primal.(a), dims)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array, ::Type{D}, dims::Dims{N}) where {P,T,D<:Dual{P,T},N}
    p_out = similar(primal.(a), P, dims)
    return Dual.(p_out, empty(T))
end

#! format: on
