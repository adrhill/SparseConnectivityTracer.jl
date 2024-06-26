#! format: off

## Type conversions (non-dual)
Base.promote_rule(::Type{T}, ::Type{N}) where {T<:AbstractTracer,N<:Real} = T
Base.promote_rule(::Type{N}, ::Type{T}) where {T<:AbstractTracer,N<:Real} = T

Base.big(::Type{T})   where {T<:AbstractTracer} = T
Base.widen(::Type{T}) where {T<:AbstractTracer} = T
Base.float(::Type{T}) where {T<:AbstractTracer} = T

Base.convert(::Type{T}, x::Real)   where {T<:AbstractTracer} = myempty(T)
Base.convert(::Type{T}, t::T)      where {T<:AbstractTracer} = t
Base.convert(::Type{<:Real}, t::T) where {T<:AbstractTracer} = t

## Constants
# These are methods defined on types. Methods on variables are in operators.jl 
Base.zero(::Type{T})        where {T<:AbstractTracer} = myempty(T)
Base.one(::Type{T})         where {T<:AbstractTracer} = myempty(T)
Base.oneunit(::Type{T})     where {T<:AbstractTracer} = myempty(T)
Base.typemin(::Type{T})     where {T<:AbstractTracer} = myempty(T)
Base.typemax(::Type{T})     where {T<:AbstractTracer} = myempty(T)
Base.eps(::Type{T})         where {T<:AbstractTracer} = myempty(T)
Base.floatmin(::Type{T})    where {T<:AbstractTracer} = myempty(T)
Base.floatmax(::Type{T})    where {T<:AbstractTracer} = myempty(T)
Base.maxintfloat(::Type{T}) where {T<:AbstractTracer} = myempty(T)

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
Base.float(::Type{D}) where {P,T,D<:Dual{P,T}} = Dual{float(P),T}

Base.convert(::Type{D}, x::Real) where {P,T,D<:Dual{P,T}}           = Dual(x, myempty(T))
Base.convert(::Type{D}, d::D)    where {P,T,D<:Dual{P,T}}           = d
Base.convert(::Type{N}, d::D)    where {N<:Real,P,T,D<:Dual{P,T}}   = Dual(convert(T, primal(d)), tracer(d))

function Base.convert(::Type{Dual{P1,T}}, d::Dual{P2,T}) where {P1,P2,T} 
    return Dual(convert(P1, primal(d)), tracer(d))
end

## Constants
# These are methods defined on types. Methods on variables are in operators.jl 
Base.zero(::Type{D})        where {P,T,D<:Dual{P,T}} = D(zero(P),        myempty(T))
Base.one(::Type{D})         where {P,T,D<:Dual{P,T}} = D(one(P),         myempty(T))
Base.oneunit(::Type{D})     where {P,T,D<:Dual{P,T}} = D(oneunit(P),     myempty(T))
Base.typemin(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemin(P),     myempty(T))
Base.typemax(::Type{D})     where {P,T,D<:Dual{P,T}} = D(typemax(P),     myempty(T))
Base.eps(::Type{D})         where {P,T,D<:Dual{P,T}} = D(eps(P),         myempty(T))
Base.floatmin(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmin(P),    myempty(T))
Base.floatmax(::Type{D})    where {P,T,D<:Dual{P,T}} = D(floatmax(P),    myempty(T))
Base.maxintfloat(::Type{D}) where {P,T,D<:Dual{P,T}} = D(maxintfloat(P), myempty(T))

#! format: on
