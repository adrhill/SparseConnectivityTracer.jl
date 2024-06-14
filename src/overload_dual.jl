
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
)
    @eval Base.$fn(d::D) where {D<:Dual} = $fn(d.primal)
    @eval function Base.$fn(t::T) where {T<:AbstractTracer}
        throw(MissingPrimalError($fn, t))
    end
end

for fn in (:isequal, :isapprox, :isless, :(==), :(<), :(>), :(<=), :(>=))
    @eval Base.$fn(dx::D, dy::D) where {D<:Dual} = $fn(dx.primal, dy.primal)
    @eval Base.$fn(dx::D, y::Real) where {D<:Dual} = $fn(dx.primal, y)
    @eval Base.$fn(x::Real, dy::D) where {D<:Dual} = $fn(x, dy.primal)
end
