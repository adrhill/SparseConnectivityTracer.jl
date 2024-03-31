## Extent Base operators
for fn in (:+, :-, :*, :/)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    for T in (:Number,)
        @eval Base.$fn(t::Tracer, ::$T) = t
        @eval Base.$fn(::$T, t::Tracer) = t
    end
end

Base.:^(a::Tracer, b::Tracer) = Tracer(a, b)
for T in (:Number, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

## Two-argument functions
for fn in (:div, :fld, :cld)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

## Single-argument functions 

#! format: off
scalar_operations = (
    :exp2, :deg2rad, :rad2deg,
    :sincos, :sincospi,
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

## Array constructors
Base.zero(::Tracer)       = Tracer()
Base.zero(::Type{Tracer}) = Tracer()
Base.one(::Tracer)        = Tracer()
Base.one(::Type{Tracer})  = Tracer()

Base.convert(::Type{Tracer}, x::Number) = Tracer()
Base.convert(::Type{Tracer}, t::Tracer) = t

Base.similar(a::Array{Tracer,1})                               = zeros(Tracer, size(a, 1))
Base.similar(a::Array{Tracer,2})                               = zeros(Tracer, size(a, 1), size(a, 2))
Base.similar(a::Array{T,1}, ::Type{Tracer}) where {T}          = zeros(Tracer, size(a, 1))
Base.similar(a::Array{T,2}, ::Type{Tracer}) where {T}          = zeros(Tracer, size(a, 1), size(a, 2))
Base.similar(::Array{Tracer}, m::Int)                          = zeros(Tracer, m)
Base.similar(::Array, ::Type{Tracer}, dims::Dims{N}) where {N} = zeros(Tracer, dims)
Base.similar(::Array{Tracer}, dims::Dims{N}) where {N}         = zeros(Tracer, dims)

## Random numbers
rand(::AbstractRNG, ::SamplerType{Tracer}) = Tracer()
