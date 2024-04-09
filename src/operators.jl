## Operator definitions

#! format: off
ops_2_to_1 = (
    :+, :-, :*, :/, 
    # division
    :div, :fld, :cld, 
    # modulo
    :mod, :rem,
    # exponentials
    :ldexp, 
    # sign
    :copysign, :flipsign,
    # other
    :hypot,
)

ops_1_to_1 = (
    # trigonometric functions
    :deg2rad, :rad2deg,
    :cos, :cosd, :cosh, :cospi, :cosc, 
    :sin, :sind, :sinh, :sinpi, :sinc, 
    :tan, :tand, :tanh,
    # reciprocal trigonometric functions
    :csc, :cscd, :csch, 
    :sec, :secd, :sech, 
    :cot, :cotd, :coth,
    # inverse trigonometric functions
    :acos, :acosd, :acosh, 
    :asin, :asind, :asinh, 
    :atan, :atand, :atanh, 
    :asec, :asech, 
    :acsc, :acsch, 
    :acot, :acoth,
    # exponentials
    :exp, :exp2, :exp10, :expm1, 
    :log, :log2, :log10, :log1p, 
    :abs, :abs2, 
    # roots
    :sqrt, :cbrt,
    # absolute values
    :abs, :abs2,
    # rounding
    :floor, :ceil, :trunc,
    # other
    :inv, :signbit, :hypot, :sign, :mod2pi
)

ops_1_to_2 = (
    # trigonometric
    :sincos,
    :sincosd,
    :sincospi,
    # exponentials
    :frexp,
)
#! format: on

for fn in ops_1_to_1
    @eval Base.$fn(t::Tracer) = t
end

for fn in ops_1_to_2
    @eval Base.$fn(t::Tracer) = (t, t)
end

for fn in ops_2_to_1
    @eval Base.$fn(a::Tracer, b::Tracer) = tracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

# Extra types required for exponent
Base.:^(a::Tracer, b::Tracer) = tracer(a, b)
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

## Precision operators create empty Tracer
for fn in (:eps, :nextfloat, :floatmin, :floatmax, :maxintfloat, :typemax)
    @eval Base.$fn(::Tracer) = tracer()
end

## Rounding
Base.round(t::Tracer, ::RoundingMode; kwargs...) = t

## Random numbers
rand(::AbstractRNG, ::SamplerType{Tracer}) = tracer()
