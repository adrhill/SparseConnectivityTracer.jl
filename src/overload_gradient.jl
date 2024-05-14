for fn in union(ops_1_to_1_s, ops_1_to_1_f)
    @eval Base.$fn(t::GlobalGradientTracer) = t
end

for fn in union(ops_1_to_1_z, ops_1_to_1_const)
    @eval Base.$fn(::T) where {T<:GlobalGradientTracer} = empty(T)
end

for fn in union(
    ops_2_to_1_ssc,
    ops_2_to_1_ssz,
    ops_2_to_1_sfc,
    ops_2_to_1_sfz,
    ops_2_to_1_fsc,
    ops_2_to_1_fsz,
    ops_2_to_1_ffc,
    ops_2_to_1_ffz,
)
    @eval Base.$fn(a::T, b::T) where {T<:GlobalGradientTracer} = T(a.grad ∪ b.grad)
    @eval Base.$fn(t::GlobalGradientTracer, ::Number) = t
    @eval Base.$fn(::Number, t::GlobalGradientTracer) = t
end

for fn in union(ops_2_to_1_zsz, ops_2_to_1_zfz)
    @eval Base.$fn(::GlobalGradientTracer, t::GlobalGradientTracer) = t
    @eval Base.$fn(::T, ::Number) where {T<:GlobalGradientTracer} = empty(T)
    @eval Base.$fn(::Number, t::GlobalGradientTracer) = t
end
for fn in union(ops_2_to_1_szz, ops_2_to_1_fzz)
    @eval Base.$fn(t::GlobalGradientTracer, ::GlobalGradientTracer) = t
    @eval Base.$fn(t::GlobalGradientTracer, ::Number) = t
    @eval Base.$fn(::Number, ::T) where {T<:GlobalGradientTracer} = empty(T)
end
for fn in ops_2_to_1_zzz
    @eval Base.$fn(::T, ::T) where {T<:GlobalGradientTracer}      = empty(T)
    @eval Base.$fn(::T, ::Number) where {T<:GlobalGradientTracer} = empty(T)
    @eval Base.$fn(::Number, ::T) where {T<:GlobalGradientTracer} = empty(T)
end

for fn in union(ops_1_to_2_ss, ops_1_to_2_sf, ops_1_to_2_fs, ops_1_to_2_ff)
    @eval Base.$fn(t::GlobalGradientTracer) = (t, t)
end

for fn in union(ops_1_to_2_sz, ops_1_to_2_fz)
    @eval Base.$fn(t::T) where {T<:GlobalGradientTracer} = (t, empty(T))
end

for fn in union(ops_1_to_2_zs, ops_1_to_2_zf)
    @eval Base.$fn(t::T) where {T<:GlobalGradientTracer} = (empty(T), t)
end
for fn in ops_1_to_2_zz
    @eval Base.$fn(::T) where {T<:GlobalGradientTracer} = (empty(T), empty(T))
end

# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::GlobalGradientTracer, ::$T) = t
    @eval Base.:^(::$T, t::GlobalGradientTracer) = t
end
Base.:^(t::GlobalGradientTracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::GlobalGradientTracer) = t

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GlobalGradientTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:GlobalGradientTracer} = empty(T)
