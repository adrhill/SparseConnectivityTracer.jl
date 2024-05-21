## 1-to-1

function overload_connectivity_1_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval $ms.$fns(t::ConnectivityTracer) = t
    @eval $ms.$fns(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($ms.$fns(primal(d)), tracer(d))
end

## 2-to-1

function overload_connectivity_2_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval $ms.$fns(a::T, b::T) where {T<:ConnectivityTracer} = T(inputs(a) ∪ inputs(b))
    @eval $ms.$fns(da::D, db::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($ms.$fns(primal(da), primal(db)), $ms.$fns(tracer(da), tracer(db)))

    @eval $ms.$fns(t::ConnectivityTracer, ::Number) = t
    @eval $ms.$fns(dx::D, y::Number) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($ms.$fns(primal(dx), y), tracer(dx))

    @eval $ms.$fns(::Number, t::ConnectivityTracer) = t
    @eval $ms.$fns(x::Number, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}} =
        Dual($ms.$fns(x, primal(dy)), tracer(dy))
end

## 1-to-2

function overload_connectivity_1_to_2(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval $ms.$fns(t::ConnectivityTracer) = (t, t)
    @eval function $ms.$fns(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        p1, p2 = $ms.$fns(primal(d))
        return (Dual(p1, tracer(d)), Dual(p2, tracer(d)))
    end
end

## Actual overloads

for op in ops_1_to_1
    overload_connectivity_1_to_1(Base, op)
end

for op in ops_2_to_1
    overload_connectivity_2_to_1(Base, op)
end

for op in ops_1_to_2
    overload_connectivity_1_to_2(Base, op)
end

## Special cases

## Exponent (requires extra types)

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
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = empty(T)  # TODO: was missing Base, add tests
