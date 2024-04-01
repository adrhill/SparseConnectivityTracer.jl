## Extent Base operators
for fn in (:+, :-, :*, :/)
    @eval Base.$fn(a::Tracer, b::Tracer) = tracer(a, b)
    for T in (:Number,)
        @eval Base.$fn(t::Tracer, ::$T) = t
        @eval Base.$fn(::$T, t::Tracer) = t
    end
end

Base.:^(a::Tracer, b::Tracer) = tracer(a, b)
for T in (:Number, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

## Two-argument functions
for fn in (:div, :fld, :cld)
    @eval Base.$fn(a::Tracer, b::Tracer) = tracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

## Single-argument functions 

#! format: off
scalar_operations = (
    :exp2, :deg2rad, :rad2deg,
    :cos,  :cosd,  :cosh, :cospi, :cosc,
    :sin,  :sind,  :sinh, :sinpi, :sinc,
    :tan,  :tand,  :tanh,
    :csc,  :cscd,  :csch,
    :sec,  :secd,  :sech,
    :cot,  :cotd,  :coth,
    :acos, :acosd, :acosh,
    :asin, :asind, :asinh,
    :atan, :atand, :atanh,
    :asec, :asech,
    :acsc, :acsch,
    :acot, :acoth,
    :exp, :expm1, :exp10,
    :frexp, :ldexp,
    :abs, :abs2, :sqrt
)
#! format: on

for fn in scalar_operations
    @eval Base.$fn(t::Tracer) = t
end

## Random numbers
rand(::AbstractRNG, ::SamplerType{Tracer}) = tracer()
