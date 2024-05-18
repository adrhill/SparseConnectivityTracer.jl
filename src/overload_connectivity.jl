## 1-to-1
for fn in nameof.(ops_1_to_1)
    @eval Base.$fn(t::ConnectivityTracer) = t
    @eval function Base.$fn(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual($fn(primal(d)), tracer(d))
    end
end

## 2-to-1
for fn in nameof.(ops_2_to_1)
    @eval Base.$fn(a::T, b::T) where {T<:ConnectivityTracer} = T(inputs(a) ∪ inputs(b))
    @eval function Base.$fn(da::D, db::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual($fn(primal(da), primal(db)), $fn(tracer(da), tracer(db)))
    end

    @eval Base.$fn(t::ConnectivityTracer, ::Number) = t
    @eval Base.$fn(dx::D, y::Number) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($fn(primal(dx), y), tracer(dx))

    @eval Base.$fn(::Number, t::ConnectivityTracer) = t
    @eval Base.$fn(x::Number, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($fn(x, primal(dy)), tracer(dy))
end

## 1-to-2
for fn in nameof.(ops_1_to_2)
    @eval Base.$fn(t::ConnectivityTracer) = (t, t)
    @eval function Base.$fn(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        p1, p2 = $fn(primal(d))
        return (Dual(p1, tracer(d)), Dual(p2, tracer(d)))
    end
end

# Extra types required for exponent
for S in (Real, Integer, Rational, Irrational{:ℯ})
    Base.:^(t::ConnectivityTracer, ::S) = t
    function Base.:^(dx::D, y::S) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual(primal(dx)^y, tracer(dx))
    end

    Base.:^(::S, t::ConnectivityTracer) = t
    function Base.:^(x::S, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual(x^primal(dy), tracer(dy))
    end
end

## Rounding
Base.round(t::ConnectivityTracer, ::RoundingMode; kwargs...) = t

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = empty(T)
