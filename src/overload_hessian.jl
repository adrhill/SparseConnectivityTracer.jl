## 1-to-1
for fn in ops_1_to_1_s
    @eval Base.$fn(t::GlobalHessianTracer) = promote_order(t)
end
for fn in ops_1_to_1_f
    @eval Base.$fn(t::GlobalHessianTracer) = t
end

for fn in union(ops_1_to_1_z, ops_1_to_1_const)
    @eval Base.$fn(::T) where {T<:GlobalHessianTracer} = empty(T)
end

## 2-to-1
# Including second-order only
for fn in ops_2_to_1_ssc
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        a = promote_order(a)
        b = promote_order(b)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_ssz
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        a = promote_order(a)
        b = promote_order(b)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = promote_order(t)
end

# Including second- and first-order
for fn in ops_2_to_1_sfc
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        a = promote_order(a)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = t
end

for fn in ops_2_to_1_sfz
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        a = promote_order(a)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = t
end

for fn in ops_2_to_1_fsc
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        b = promote_order(b)
        return distributive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_fsz
    @eval function Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer)
        b = promote_order(b)
        return additive_merge(a, b)
    end
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = promote_order(t)
end

# Including first-order only
for fn in ops_2_to_1_ffc
    @eval Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer) =
        distributive_merge(a, b)
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = t
end

for fn in ops_2_to_1_ffz
    @eval Base.$fn(a::GlobalHessianTracer, b::GlobalHessianTracer) = additive_merge(a, b)
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = t
end

# Including zero-order
for fn in ops_2_to_1_szz
    @eval Base.$fn(t::GlobalHessianTracer, ::GlobalHessianTracer) = promote_order(t)
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = promote_order(t)
    @eval Base.$fn(::Number, t::T) where {T<:GlobalHessianTracer} = empty(T)
end

for fn in ops_2_to_1_zsz
    @eval Base.$fn(::GlobalHessianTracer, t::GlobalHessianTracer) = promote_order(t)
    @eval Base.$fn(::T, ::Number) where {T<:GlobalHessianTracer} = empty(T)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = promote_order(t)
end

for fn in ops_2_to_1_fzz
    @eval Base.$fn(t::GlobalHessianTracer, ::GlobalHessianTracer) = t
    @eval Base.$fn(t::GlobalHessianTracer, ::Number) = t
    @eval Base.$fn(::Number, t::T) where {T<:GlobalHessianTracer} = empty(T)
end

for fn in ops_2_to_1_zfz
    @eval Base.$fn(::GlobalHessianTracer, t::GlobalHessianTracer) = t
    @eval Base.$fn(::T, ::Number) where {T<:GlobalHessianTracer} = empty(T)
    @eval Base.$fn(::Number, t::GlobalHessianTracer) = t
end

for fn in ops_2_to_1_zzz
    @eval Base.$fn(::T, t::T) where {T<:GlobalHessianTracer} = empty(T)
    @eval Base.$fn(::T, ::Number) where {T<:GlobalHessianTracer} = empty(T)
    @eval Base.$fn(::Number, t::T) where {T<:GlobalHessianTracer} = empty(T)
end

# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::GlobalHessianTracer, ::$T) = promote_order(t)
    @eval Base.:^(::$T, t::GlobalHessianTracer) = promote_order(t)
end
Base.:^(t::GlobalHessianTracer, ::Irrational{:ℯ}) = promote_order(t)
Base.:^(::Irrational{:ℯ}, t::GlobalHessianTracer) = promote_order(t)

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GlobalHessianTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:GlobalHessianTracer} = empty(T)
