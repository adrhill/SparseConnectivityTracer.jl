#! format: off

## Type conversions (non-dual)
for TT in (GradientTracer, ConnectivityTracer, HessianTracer)
    Base.promote_rule(::Type{T}, ::Type{N}) where {T<:TT,N<:Number} = T
    Base.promote_rule(::Type{N}, ::Type{T}) where {T<:TT,N<:Number} = T

    Base.big(::Type{T})   where {T<:TT} = T
    Base.widen(::Type{T}) where {T<:TT} = T

    Base.convert(::Type{T}, x::Number)   where {T<:TT} = empty(T)
    Base.convert(::Type{T}, t::T)        where {T<:TT} = t
    Base.convert(::Type{<:Number}, t::T) where {T<:TT} = t

    ## Constants
    Base.zero(::Type{T})        where {T<:TT} = empty(T)
    Base.one(::Type{T})         where {T<:TT} = empty(T)
    Base.oneunit(::Type{T})     where {T<:TT} = empty(T)
    Base.typemin(::Type{T})     where {T<:TT} = empty(T)
    Base.typemax(::Type{T})     where {T<:TT} = empty(T)
    Base.eps(::Type{T})         where {T<:TT} = empty(T)
    Base.float(::Type{T})       where {T<:TT} = empty(T)
    Base.floatmin(::Type{T})    where {T<:TT} = empty(T)
    Base.floatmax(::Type{T})    where {T<:TT} = empty(T)
    Base.maxintfloat(::Type{T}) where {T<:TT} = empty(T)
    
    ## Array constructors
    Base.similar(a::Array{T,1})                     where {T<:TT}   = zeros(T, size(a, 1))
    Base.similar(a::Array{T,2})                     where {T<:TT}   = zeros(T, size(a, 1), size(a, 2))
    Base.similar(a::Array{A,1}, ::Type{T})          where {T<:TT,A} = zeros(T, size(a, 1))
    Base.similar(a::Array{A,2}, ::Type{T})          where {T<:TT,A} = zeros(T, size(a, 1), size(a, 2))
    Base.similar(::Array{T}, m::Int)                where {T<:TT}   = zeros(T, m)
    Base.similar(::Array{T}, dims::Dims{N})         where {T<:TT,N} = zeros(T, dims)
end

Base.similar(::Array, ::Type{ConnectivityTracer{C}}, dims::Dims{N}) where {C,N}   = zeros(ConnectivityTracer{C}, dims)
Base.similar(::Array, ::Type{GradientTracer{G}},     dims::Dims{N}) where {G,N}   = zeros(GradientTracer{G},     dims)
Base.similar(::Array, ::Type{HessianTracer{G,H}},    dims::Dims{N}) where {G,H,N} = zeros(HessianTracer{G,H},    dims)

## Duals
function Base.promote_rule(::Type{Dual{P1, T}}, ::Type{Dual{P2, T}}) where {P1,P2,T}
    PP = Base.promote_type(P1, P2) # TODO: possible method call error?
    return Dual{PP,T}
end
function Base.promote_rule(::Type{Dual{P, T}}, ::Type{N}) where {P,T,N<:Number}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP,T}
end
function Base.promote_rule(::Type{N}, ::Type{Dual{P, T}}) where {P,T,N<:Number}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP,T}
end

Base.big(::Type{D})   where {P,T,D<:Dual{P,T}} = Dual{big(P),T}
Base.widen(::Type{D}) where {P,T,D<:Dual{P,T}} = Dual{widen(P),T}

Base.convert(::Type{D}, x::Number) where {P,T,D<:Dual{P,T}}           = Dual(x, empty(T))
Base.convert(::Type{D}, d::D)      where {P,T,D<:Dual{P,T}}           = d
Base.convert(::Type{N}, d::D)      where {N<:Number,P,T,D<:Dual{P,T}} = Dual(convert(T, primal(d)), tracer(d))

function Base.convert(::Type{Dual{P1,T}}, d::Dual{P2,T}) where {P1,P2,T} 
    return Dual(convert(P1, primal(d)), tracer(d))
end

## Constants
Base.zero(::Type{D})        where {P,T,D<:Dual{P,T}} = D(zero(P),        empty(T))
Base.one(::Type{D})         where {P,T,D<:Dual{P,T}} = D(one(P),         empty(T))
Base.oneunit(::Type{D})     where {P,T,D<:Dual{P,T}} = D(oneunit(P),     empty(T))
Base.typemin(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemin(P),     empty(T))
Base.typemax(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemax(P),     empty(T))
Base.eps(::Type{D})         where {P,T,D<:Dual{P,T}} = D(eps(P),         empty(T))
Base.float(::Type{D})       where {P,T,D<:Dual{P,T}} = D(float(P),       empty(T))
Base.floatmin(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmin(P),    empty(T))
Base.floatmax(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmax(P),    empty(T))
Base.maxintfloat(::Type{D}) where {P,T,D<:Dual{P,T}} = D(maxintfloat(P), empty(T))

## Array constructors
function Base.similar(a::Array{D,1}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D,2}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{A,1}, ::Type{D}) where {P,T,D<:Dual{P,T},A}
    p_out = similar(a, P)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{A,2}, ::Type{D}) where {P,T,D<:Dual{P,T},A}
    p_out = similar(a, P)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D}, m::Int) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a), m)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array{D}, dims::Dims{N}) where {P,T,D<:Dual{P,T}, N}
    p_out = similar(primal.(a), dims)
    return Dual.(p_out, empty(T))
end
function Base.similar(a::Array, ::Type{Dual{P,T}}, dims::Dims{N}) where {P,T,N}
    p_out = similar(primal.(a), P, dims)
    return Dual.(p_out, empty(T))
end

#! format: on
