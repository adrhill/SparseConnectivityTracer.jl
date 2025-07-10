## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:AbstractTracer} = myempty(T)

function Base.round(
    d::D, mode::RoundingMode; kwargs...
) where {P,T<:AbstractTracer,D<:Dual{P,T}}
    return round(primal(d), mode; kwargs...) # only return primal
end

for RR in (Real, Integer, Bool)
    Base.round(::Type{R}, ::T) where {R<:RR,T<:AbstractTracer} = myempty(T)
    function Base.round(::Type{R}, d::D) where {R<:RR,P,T<:AbstractTracer,D<:Dual{P,T}}
        return round(R, primal(d)) # only return primal
    end
end

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:AbstractTracer} = myempty(T)
function Base.rand(
    rng::AbstractRNG, ::SamplerType{D}
) where {P,T<:AbstractTracer,D<:Dual{P,T}}
    p = rand(rng, P)
    # This unfortunately can't just return the primal value.
    # Random.jl will otherwise throw "TypeError: in typeassert, expected Dual{P,T}, got a value of type P".
    t = myempty(T)
    return Dual(p, t)
end
