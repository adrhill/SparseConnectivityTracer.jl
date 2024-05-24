#! format: off

## Type conversions (non-dual)
for TT in (GradientTracer, ConnectivityTracer, HessianTracer)
    Base.promote_rule(::Type{T}, ::Type{N}) where {T<:TT,N<:Real} = T
    Base.promote_rule(::Type{N}, ::Type{T}) where {T<:TT,N<:Real} = T

    Base.big(::Type{T})   where {T<:TT} = T
    Base.widen(::Type{T}) where {T<:TT} = T

    Base.convert(::Type{T}, x::Real)   where {T<:TT} = myempty(T)
    Base.convert(::Type{T}, t::T)        where {T<:TT} = t
    Base.convert(::Type{<:Real}, t::T) where {T<:TT} = t

    ## Constants
    Base.zero(::Type{T})        where {T<:TT} = myempty(T)
    Base.one(::Type{T})         where {T<:TT} = myempty(T)
    Base.oneunit(::Type{T})     where {T<:TT} = myempty(T)
    Base.typemin(::Type{T})     where {T<:TT} = myempty(T)
    Base.typemax(::Type{T})     where {T<:TT} = myempty(T)
    Base.eps(::Type{T})         where {T<:TT} = myempty(T)
    Base.float(::Type{T})       where {T<:TT} = myempty(T)
    Base.floatmin(::Type{T})    where {T<:TT} = myempty(T)
    Base.floatmax(::Type{T})    where {T<:TT} = myempty(T)
    Base.maxintfloat(::Type{T}) where {T<:TT} = myempty(T)
    
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
function Base.promote_rule(::Type{Dual{P, T}}, ::Type{N}) where {P,T,N<:Real}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP,T}
end
function Base.promote_rule(::Type{N}, ::Type{Dual{P, T}}) where {P,T,N<:Real}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP,T}
end

Base.big(::Type{D})   where {P,T,D<:Dual{P,T}} = Dual{big(P),T}
Base.widen(::Type{D}) where {P,T,D<:Dual{P,T}} = Dual{widen(P),T}

Base.convert(::Type{D}, x::Real) where {P,T,D<:Dual{P,T}}           = Dual(x, myempty(T))
Base.convert(::Type{D}, d::D)      where {P,T,D<:Dual{P,T}}           = d
Base.convert(::Type{N}, d::D)      where {N<:Real,P,T,D<:Dual{P,T}} = Dual(convert(T, primal(d)), tracer(d))

function Base.convert(::Type{Dual{P1,T}}, d::Dual{P2,T}) where {P1,P2,T} 
    return Dual(convert(P1, primal(d)), tracer(d))
end

## Constants
Base.zero(::Type{D})        where {P,T,D<:Dual{P,T}} = D(zero(P),        myempty(T))
Base.one(::Type{D})         where {P,T,D<:Dual{P,T}} = D(one(P),         myempty(T))
Base.oneunit(::Type{D})     where {P,T,D<:Dual{P,T}} = D(oneunit(P),     myempty(T))
Base.typemin(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemin(P),     myempty(T))
Base.typemax(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemax(P),     myempty(T))
Base.eps(::Type{D})         where {P,T,D<:Dual{P,T}} = D(eps(P),         myempty(T))
Base.float(::Type{D})       where {P,T,D<:Dual{P,T}} = D(float(P),       myempty(T))
Base.floatmin(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmin(P),    myempty(T))
Base.floatmax(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmax(P),    myempty(T))
Base.maxintfloat(::Type{D}) where {P,T,D<:Dual{P,T}} = D(maxintfloat(P), myempty(T))

## Array constructors
function Base.similar(a::Array{D,1}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{D,2}) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a))
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{A,1}, ::Type{D}) where {P,T,D<:Dual{P,T},A}
    p_out = similar(a, P)
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{A,2}, ::Type{D}) where {P,T,D<:Dual{P,T},A}
    p_out = similar(a, P)
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{D}, m::Int) where {P,T,D<:Dual{P,T}}
    p_out = similar(primal.(a), m)
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{D}, dims::Dims{N}) where {P,T,D<:Dual{P,T}, N}
    p_out = similar(primal.(a), dims)
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array{D2}, ::Type{Dual{P,T}}, dims::Dims{N}) where {P,T,N,D2<:Dual}
    p_out = similar(primal.(a), P, dims)
    return Dual.(p_out, myempty(T))
end
function Base.similar(a::Array, ::Type{Dual{P,T}}, dims::Dims{N}) where {P,T,N}
    p_out = similar(a, P, dims)
    return Dual.(p_out, myempty(T))
end

#! format: on
