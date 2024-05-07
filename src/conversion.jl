## Type conversions
for TT in (:JacobianTracer, :ConnectivityTracer, :HessianTracer)
    @eval Base.promote_rule(::Type{T}, ::Type{N}) where {T<:$TT,N<:Number} = T
    @eval Base.promote_rule(::Type{N}, ::Type{T}) where {T<:$TT,N<:Number} = T

    @eval Base.big(::Type{T}) where {T<:$TT}   = T
    @eval Base.widen(::Type{T}) where {T<:$TT} = T
    @eval Base.widen(t::T) where {T<:$TT}      = t

    @eval Base.convert(::Type{T}, x::Number) where {T<:$TT}   = empty(T)
    @eval Base.convert(::Type{T}, t::T) where {T<:$TT}        = t
    @eval Base.convert(::Type{<:Number}, t::T) where {T<:$TT} = t

    ## Constants
    @eval Base.zero(::Type{T}) where {T<:$TT}    = empty(T)
    @eval Base.one(::Type{T}) where {T<:$TT}     = empty(T)
    @eval Base.typemin(::Type{T}) where {T<:$TT} = empty(T)
    @eval Base.typemax(::Type{T}) where {T<:$TT} = empty(T)

    ## Array constructors
    @eval Base.similar(a::Array{T,1}) where {T<:$TT}               = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{T,2}) where {T<:$TT}               = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(a::Array{A,1}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1))
    @eval Base.similar(a::Array{A,2}, ::Type{T}) where {A,T<:$TT}  = zeros(T, size(a, 1), size(a, 2))
    @eval Base.similar(::Array{T}, m::Int) where {T<:$TT}          = zeros(T, m)
    @eval Base.similar(::Array{T}, dims::Dims{N}) where {N,T<:$TT} = zeros(T, dims)

    @eval Base.similar(::Array, ::Type{$TT{S}}, dims::Dims{N}) where {N,S} =
        zeros($TT{S}, dims)
end

Base.real(sct::SparseConnectivityTracer.AbstractTracer) = sct
Base.float(sct::SparseConnectivityTracer.AbstractTracer) = sct
Base.max(x::SparseConnectivityTracer.AbstractTracer, y::SparseConnectivityTracer.AbstractTracer) = x
Base.floatmin(x::Type{<:SparseConnectivityTracer.AbstractTracer}) = Base.floatmin(Float32)
Base.eps(x::Type{<:SparseConnectivityTracer.AbstractTracer}) = Base.eps(Float32)
Base.isinf(::SparseConnectivityTracer.AbstractTracer) = false
Base.isless(x::SparseConnectivityTracer.AbstractTracer, ::SparseConnectivityTracer.AbstractTracer) = true
Base.isless(::SparseConnectivityTracer.AbstractTracer, ::Float64) = true
Base.isfinite(::SparseConnectivityTracer.AbstractTracer) = true
Base.conj(x::SparseConnectivityTracer.AbstractTracer) = x

