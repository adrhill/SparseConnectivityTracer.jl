
# Special overloads for Dual numbers
for fn in (
    :iseven,
    :isfinite,
    :isinf,
    :isinteger,
    :ismissing,
    :isnan,
    :isnothing,
    :isodd,
    :isone,
    :isreal,
    :iszero,
    :real,
)
    @eval Base.$fn(d::D) where {D<:Dual} = $fn(primal(d))
    @eval function Base.$fn(t::T) where {T<:AbstractTracer}
        throw(MissingPrimalError($fn, t))
    end
end

for fn in (:isequal, :isapprox, :isless, :(==), :(<), :(>), :(<=), :(>=))
    @eval Base.$fn(dx::D, dy::D) where {D<:Dual} = $fn(primal(dx), primal(dy))
    @eval Base.$fn(dx::D, y::Real) where {D<:Dual} = $fn(primal(dx), y)
    @eval Base.$fn(x::Real, dy::D) where {D<:Dual} = $fn(x, primal(dy))

    # Error on non-dual tracers
    @eval function Base.$fn(tx::T, ty::T) where {T<:AbstractTracer}
        return throw(MissingPrimalError($fn, tx))
    end
    @eval function Base.$fn(tx::T, y::Real) where {T<:AbstractTracer}
        return throw(MissingPrimalError($fn, tx))
    end
    @eval function Base.$fn(x::Real, ty::T) where {T<:AbstractTracer}
        return throw(MissingPrimalError($fn, ty))
    end
end
