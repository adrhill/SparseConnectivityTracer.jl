## Type conversions
Base.convert(::Type{Tracer}, x::Number)   = Tracer()
Base.convert(::Type{Tracer}, t::Tracer)   = t
Base.convert(::Type{<:Number}, t::Tracer) = t

Base.promote_rule(::Type{Tracer}, ::Type{N}) where {N<:Number} = Tracer
Base.promote_rule(::Type{N}, ::Type{Tracer}) where {N<:Number} = Tracer

## Array constructors
Base.zero(::Tracer)       = Tracer()
Base.zero(::Type{Tracer}) = Tracer()
Base.one(::Tracer)        = Tracer()
Base.one(::Type{Tracer})  = Tracer()

Base.similar(a::Array{Tracer,1})                               = zeros(Tracer, size(a, 1))
Base.similar(a::Array{Tracer,2})                               = zeros(Tracer, size(a, 1), size(a, 2))
Base.similar(a::Array{T,1}, ::Type{Tracer}) where {T}          = zeros(Tracer, size(a, 1))
Base.similar(a::Array{T,2}, ::Type{Tracer}) where {T}          = zeros(Tracer, size(a, 1), size(a, 2))
Base.similar(::Array{Tracer}, m::Int)                          = zeros(Tracer, m)
Base.similar(::Array, ::Type{Tracer}, dims::Dims{N}) where {N} = zeros(Tracer, dims)
Base.similar(::Array{Tracer}, dims::Dims{N}) where {N}         = zeros(Tracer, dims)
