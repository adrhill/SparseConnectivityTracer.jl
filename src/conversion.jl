## Type conversions
Base.promote_rule(::Type{T}, ::Type{N}) where {N<:Number,T<:AbstractTracer} = T
Base.promote_rule(::Type{N}, ::Type{T}) where {N<:Number,T<:AbstractTracer} = T

Base.big(::Type{T}) where {T<:AbstractTracer}   = T
Base.widen(::Type{T}) where {T<:AbstractTracer} = T
Base.widen(t::AbstractTracer)                   = t

Base.convert(::Type{T}, x::Number) where {T<:AbstractTracer} = empty(T)
Base.convert(::Type{T}, t::T) where {T<:AbstractTracer} = t
Base.convert(::Type{<:Number}, t::AbstractTracer) = t

## Array constructors
Base.zero(::Type{T}) where {T<:AbstractTracer} = empty(T)
Base.one(::Type{T}) where {T<:AbstractTracer}  = empty(T)

Base.similar(a::Array{T,1}) where {T<:AbstractTracer}                       = zeros(T, size(a, 1))
Base.similar(a::Array{T,2}) where {T<:AbstractTracer}                       = zeros(T, size(a, 1), size(a, 2))
Base.similar(a::Array{A,1}, ::Type{T}) where {A,T<:AbstractTracer}          = zeros(T, size(a, 1))
Base.similar(a::Array{A,2}, ::Type{T}) where {A,T<:AbstractTracer}          = zeros(T, size(a, 1), size(a, 2))
Base.similar(::Array{T}, m::Int) where {T<:AbstractTracer}                  = zeros(T, m)
Base.similar(::Array, ::Type{T}, dims::Dims{N}) where {N,T<:AbstractTracer} = zeros(T, dims)
Base.similar(::Array{T}, dims::Dims{N}) where {N,T<:AbstractTracer}         = zeros(T, dims)
