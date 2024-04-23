for fn in union(ops_1_to_1_s, ops_1_to_1_f, ops_1_to_1_z)
    @eval Base.$fn(t::ConnectivityTracer) = t
end

for fn in ops_1_to_1_const
    @eval Base.$fn(::ConnectivityTracer) = EMPTY_CONNECTIVITY_TRACER
end

for fn in ops_1_to_2
    @eval Base.$fn(t::ConnectivityTracer) = (t, t)
end

for fn in ops_2_to_1
    @eval Base.$fn(a::ConnectivityTracer, b::ConnectivityTracer) = uniontracer(a, b)
    @eval Base.$fn(t::ConnectivityTracer, ::Number) = t
    @eval Base.$fn(::Number, t::ConnectivityTracer) = t
end

# Extra types required for exponent
Base.:^(a::ConnectivityTracer, b::ConnectivityTracer) = uniontracer(a, b)
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::ConnectivityTracer, ::$T) = t
    @eval Base.:^(::$T, t::ConnectivityTracer) = t
end
Base.:^(t::ConnectivityTracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::ConnectivityTracer) = t

## Rounding
Base.round(t::ConnectivityTracer, ::RoundingMode; kwargs...) = t

## Random numbers
rand(::AbstractRNG, ::SamplerType{ConnectivityTracer}) = EMPTY_CONNECTIVITY_TRACER
