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
