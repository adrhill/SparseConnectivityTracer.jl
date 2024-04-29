## 1-to-1
for fn in ops_1_to_1_s
    @eval Base.$fn(t::HessianTracer) = promote_order(t)
end
for fn in ops_1_to_1_f
    @eval Base.$fn(t::HessianTracer) = t
end

for fn in union(ops_1_to_1_z, ops_1_to_1_const)
    @eval Base.$fn(::HessianTracer) = EMPTY_HESSIAN_TRACER
end

## 2-to-1
# Including second-order only
for fn in ops_2_to_1_ssc
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        a = promote_order(a)
        b = promote_order(b)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::HessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_ssz
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        a = promote_order(a)
        b = promote_order(b)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::HessianTracer) = promote_order(t)
end

# Including second- and first-order
for fn in ops_2_to_1_sfc
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        a = promote_order(a)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::HessianTracer) = t
end

for fn in ops_2_to_1_sfz
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        a = promote_order(a)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::HessianTracer) = t
end

for fn in ops_2_to_1_fsc
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        b = promote_order(b)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::HessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_fsz
    @eval function Base.$fn(a::HessianTracer, b::HessianTracer)
        b = promote_order(b)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::HessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::HessianTracer) = promote_order(t)
end

# Including first-order only
for fn in ops_2_to_1_ffc
    @eval Base.$fn(a::HessianTracer, b::HessianTracer) = distributive_merge(a, b)
    @eval Base.$fn(t::HessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::HessianTracer) = t
end

for fn in ops_2_to_1_ffz
    @eval Base.$fn(a::HessianTracer, b::HessianTracer) = additive_merge(a, b)
    @eval Base.$fn(t::HessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::HessianTracer) = t
end

# Including zero-order
for fn in ops_2_to_1_szz
    @eval Base.$fn(t::HessianTracer, ::HessianTracer) = promote_order(t)
    @eval Base.$fn(t::HessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::HessianTracer) = EMPTY_HESSIAN_TRACER
end

for fn in ops_2_to_1_zsz
    @eval Base.$fn(::HessianTracer, t::HessianTracer) = promote_order(t)
    @eval Base.$fn(::HessianTracer, ::Number) = EMPTY_HESSIAN_TRACER
    @eval Base.$fn(::Number, t::HessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_fzz
    @eval Base.$fn(t::HessianTracer, ::HessianTracer) = t
    @eval Base.$fn(t::HessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::HessianTracer) = EMPTY_HESSIAN_TRACER
end

for fn in ops_2_to_1_zfz
    @eval Base.$fn(::HessianTracer, t::HessianTracer) = t
    @eval Base.$fn(::HessianTracer, ::Number) = EMPTY_HESSIAN_TRACER
    @eval Base.$fn(::Number, t::HessianTracer) = t
end

for fn in ops_2_to_1_zzz
    @eval Base.$fn(::HessianTracer, t::HessianTracer) = EMPTY_HESSIAN_TRACER
    @eval Base.$fn(::HessianTracer, ::Number) = EMPTY_HESSIAN_TRACER
    @eval Base.$fn(::Number, t::HessianTracer) = EMPTY_HESSIAN_TRACER
end

# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::HessianTracer, ::$T) = promote_order(t)
    @eval Base.:^(::$T, t::HessianTracer) = promote_order(t)
end
Base.:^(t::HessianTracer, ::Irrational{:ℯ}) = promote_order(t)
Base.:^(::Irrational{:ℯ}, t::HessianTracer) = promote_order(t)

## Rounding
Base.round(t::HessianTracer, ::RoundingMode; kwargs...) = EMPTY_HESSIAN_TRACER

## Random numbers
rand(::AbstractRNG, ::SamplerType{HessianTracer}) = EMPTY_HESSIAN_TRACER
