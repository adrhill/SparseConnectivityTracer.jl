## Type conversions
for T in (:JacobianTracer, :ConnectivityTracer, :HessianTracer)
    @eval Base.promote_rule(::Type{$T}, ::Type{N}) where {N<:Number} = $T
    @eval Base.promote_rule(::Type{N}, ::Type{$T}) where {N<:Number} = $T

    @eval Base.big(::Type{$T})   = $T
    @eval Base.widen(::Type{$T}) = $T
    @eval Base.widen(t::$T)      = t

    @eval Base.convert(::Type{$T}, x::Number) = empty($T)
    @eval Base.convert(::Type{$T}, t::$T) = t
    @eval Base.convert(::Type{<:Number}, t::$T) = t

    ## Constants
    @eval Base.zero(::Type{$T})    = empty($T)
    @eval Base.one(::Type{$T})     = empty($T)
    @eval Base.typemin(::Type{$T}) = empty($T)
    @eval Base.typemax(::Type{$T}) = empty($T)

    ## Array constructors
    @eval Base.similar(a::Array{$T,1})                               = zeros($T, size(a, 1))
    @eval Base.similar(a::Array{$T,2})                               = zeros($T, size(a, 1), size(a, 2))
    @eval Base.similar(a::Array{A,1}, ::Type{$T}) where {A}          = zeros($T, size(a, 1))
    @eval Base.similar(a::Array{A,2}, ::Type{$T}) where {A}          = zeros($T, size(a, 1), size(a, 2))
    @eval Base.similar(::Array{$T}, m::Int)                          = zeros($T, m)
    @eval Base.similar(::Array, ::Type{$T}, dims::Dims{N}) where {N} = zeros($T, dims)
    @eval Base.similar(::Array{$T}, dims::Dims{N}) where {N}         = zeros($T, dims)
end
