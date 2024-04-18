for fn in union(ops_1_to_1_s, ops_1_to_1_f, ops_1_to_1_z)
    @eval Base.$fn(t::Tracer) = t
end

for fn in ops_1_to_1_const
    @eval Base.$fn(::Tracer) = EMPTY_TRACER
end

for fn in ops_1_to_2
    @eval Base.$fn(t::Tracer) = (t, t)
end

for fn in ops_2_to_1
    @eval Base.$fn(a::Tracer, b::Tracer) = uniontracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

# Extra types required for exponent
Base.:^(a::Tracer, b::Tracer) = uniontracer(a, b)
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

## Rounding
Base.round(t::Tracer, ::RoundingMode; kwargs...) = t

## Random numbers
rand(::AbstractRNG, ::SamplerType{Tracer}) = EMPTY_TRACER
