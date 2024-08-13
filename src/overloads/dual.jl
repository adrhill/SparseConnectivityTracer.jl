
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
    @eval Base.$fn(d::D) where {D<:Dual} = $fn(primal(d))
    @eval function Base.$fn(t::T) where {T<:AbstractTracer}
        throw(MissingPrimalError($fn, t))
    end
end

for fn in (:isequal, :isapprox, :isless, :(==), :(<), :(>), :(<=), :(>=))
    @eval Base.$fn(dx::D, dy::D) where {D<:Dual} = $fn(primal(dx), primal(dy))
    @eval Base.$fn(dx::D, y::Real) where {D<:Dual} = $fn(primal(dx), y)
    @eval Base.$fn(x::Real, dy::D) where {D<:Dual} = $fn(x, primal(dy))
end

# In some cases, more specialized methods are needed
Base.isless(dx::D, y::AbstractFloat) where {D<:Dual} = isless(primal(dx), y)
