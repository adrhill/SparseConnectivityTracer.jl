# GradientTracer & HessianTracer
for fn in (
        :iseven,
        :isfinite,
        :isinf,
        :isreal,
        :isinteger,
        :ismissing,
        :isnan,
        :isnothing,
        :isodd,
        :isone,
        :iszero,
    )
    @eval Base.$fn(t::AbstractTracer) = throw(MissingPrimalError($fn, t))
end

# Dual
for fn in (
        :iseven,
        :isfinite,
        :isinf,
        :isinteger,
        :isnan,
        :isodd,
        :isone,
        :iszero,
    )
    @eval Base.$fn(d::Dual) = $fn(primal(d))
end
