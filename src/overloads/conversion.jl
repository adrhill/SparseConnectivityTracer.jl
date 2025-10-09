##===============#
# AbstractTracer #
#================#

Base.promote_rule(::Type{T}, ::Type{N}) where {T <: AbstractTracer, N <: Real} = T
Base.promote_rule(::Type{N}, ::Type{T}) where {T <: AbstractTracer, N <: Real} = T

Base.convert(::Type{T}, x::Real) where {T <: AbstractTracer} = myempty(T)
Base.convert(::Type{T}, t::T) where {T <: AbstractTracer} = t
Base.convert(::Type{<:Real}, t::T) where {T <: AbstractTracer} = t

##======#
# Duals #
#=======#

function Base.promote_rule(::Type{Dual{P1, T}}, ::Type{Dual{P2, T}}) where {P1, P2, T}
    PP = Base.promote_type(P1, P2) # TODO: possible method call error?
    return Dual{PP, T}
end
function Base.promote_rule(::Type{Dual{P, T}}, ::Type{N}) where {P, T, N <: Real}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP, T}
end
function Base.promote_rule(::Type{N}, ::Type{Dual{P, T}}) where {P, T, N <: Real}
    PP = Base.promote_type(P, N) # TODO: possible method call error?
    return Dual{PP, T}
end

Base.convert(::Type{D}, x::Real) where {P, T, D <: Dual{P, T}} = D(convert(P, x), myempty(T))
Base.convert(::Type{D}, d::D) where {D <: Dual} = d
Base.convert(::Type{N}, d::D) where {N <: Real, P, T, D <: Dual{P, T}} = Dual{N, T}(convert(N, primal(d)), tracer(d))

function Base.convert(::Type{Dual{P1, T}}, d::Dual{P2, T}) where {P1, P2, T}
    return Dual{P1, T}(convert(P1, primal(d)), tracer(d))
end

##==========================#
# Explicit type conversions #
#===========================#

for T in (:Int, :Integer, :Float64, :Float32)
    # Currently only defined on Dual to avoid invalidations.
    @eval function Base.$T(d::Dual)
        isemptytracer(d) || throw(InexactError(Symbol($T), $T, d))
        return $T(primal(d))
    end
end

##======================#
# Named type promotions #
#=======================#

for f in (:big, :widen, :float)
    @eval Base.$f(::Type{T}) where {T <: AbstractTracer} = T
    @eval Base.$f(::Type{D}) where {P, T, D <: Dual{P, T}} = $f(P) # only return primal type
end

##============================#
# Constant functions on types #
#=============================#

for f in
    (:zero, :one, :oneunit, :typemin, :typemax, :eps, :floatmin, :floatmax, :maxintfloat)
    @eval Base.$f(::Type{T}) where {T <: AbstractTracer} = myempty(T)
    @eval Base.$f(::Type{D}) where {P, T, D <: Dual{P, T}} = Dual($f(P), myempty(T))
    @eval Base.$f(::T) where {T <: AbstractTracer} = myempty(T)
    @eval Base.$f(d::D) where {P, T, D <: Dual{P, T}} = Dual($f(d.primal), myempty(T))
end
