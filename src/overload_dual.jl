
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
    @eval Base.$fn(dx::D, y::Number) where {D<:Dual} = $fn(primal(dx), y)
    @eval Base.$fn(x::Number, dy::D) where {D<:Dual} = $fn(x, primal(dy))
end
