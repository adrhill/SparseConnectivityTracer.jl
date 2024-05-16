# Overload comparisons on Dual numbers
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
    @eval Base.$fn(d::D) where {D<:Dual} = Base.$fn(primal(d))
end

for fn in (:isequal, :isapprox, :isless, :(==), :(<), :(>), :(<=), :(>=))
    @eval Base.$fn(dx::D, dy::D) where {D<:Dual} = Base.$fn(primal(dx), primal(dy))
end
