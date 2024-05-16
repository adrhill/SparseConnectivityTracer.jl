## 1-to-1
function gradient_tracer_1_to_1(t::T, is_firstder_zero::Bool) where {T<:GradientTracer}
    if is_firstder_zero
        return empty(T)
    else
        return t
    end
end

for fn in ops_1_to_1
    @eval function Base.$fn(t::T) where {T<:GradientTracer}
        return gradient_tracer_1_to_1(t, is_firstder_zero_global($fn))
    end
    @eval function Base.$fn(d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(d)
        p_out = Base.$fn(x)
        t_out = gradient_tracer_1_to_1(tracer(d), is_firstder_zero_local($fn, x))
        return Dual(p_out, t_out)
    end
end

## 2-to-1
function gradient_tracer_2_to_1(
    tx::T,
    ty::T,
    is_firstder_arg1_zero_or_number::Bool,
    is_firstder_arg2_zero_or_number::Bool,
) where {T<:GradientTracer}
    if is_firstder_arg1_zero_or_number
        if is_firstder_arg2_zero_or_number
            return empty(T)
        else
            return ty
        end
    else # ∂f∂x ≠ 0 
        if is_firstder_arg2_zero_or_number
            return tx
        else
            return T(gradient(tx) ∪ gradient(ty))
        end
    end
end

function gradient_tracer_2_to_1_one_tracer(
    t::T, is_firstder_zero::Bool
) where {T<:GradientTracer}
    if is_firstder_zero
        return empty(T)
    else
        return t
    end
end

for fn in ops_2_to_1
    @eval function Base.$fn(tx::T, ty::T) where {T<:GradientTracer}
        return gradient_tracer_2_to_1(
            tx, ty, is_firstder_arg1_zero_global($fn), is_firstder_arg2_zero_global($fn)
        )
    end
    @eval function Base.$fn(dx::D, dy::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(dx)
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = gradient_tracer_2_to_1(
            tracer(dx),
            tracer(dy),
            is_firstder_arg1_zero_local($fn, x, y),
            is_firstder_arg2_zero_local($fn, x, y),
        )
        return Dual(p_out, t_out)
    end

    @eval function Base.$fn(tx::T, ::Number) where {T<:GradientTracer}
        return gradient_tracer_2_to_1_one_tracer(tx, is_firstder_arg1_zero_global($fn))
    end
    @eval function Base.$fn(dx::D, y::Number) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(dx)
        p_out = Base.$fn(x, y)
        t_out = gradient_tracer_2_to_1_one_tracer(
            tracer(dx), is_firstder_arg1_zero_local($fn, x, y)
        )
        return Dual(p_out, t_out)
    end

    @eval function Base.$fn(::Number, ty::T) where {T<:GradientTracer}
        return gradient_tracer_2_to_1_one_tracer(ty, is_firstder_arg2_zero_global($fn))
    end
    @eval function Base.$fn(x::Number, dy::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = gradient_tracer_2_to_1_one_tracer(
            tracer(dx), is_firstder_arg2_zero_local($fn, x, y)
        )
        return Dual(p_out, t_out)
    end
end

## 1-to-2
function gradient_tracer_1_to_2(
    t::T, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {T<:GradientTracer}
    t1 = gradient_tracer_1_to_1(t, is_firstder_out1_zero)
    t2 = gradient_tracer_1_to_1(t, is_firstder_out2_zero)
    return (t1, t2)
end

for fn in ops_1_to_2
    @eval function Base.$fn(t::T) where {T<:GradientTracer}
        return gradient_tracer_1_to_2(
            t, is_firstder_out1_zero_global($fn), is_firstder_out2_zero_global($fn)
        )
    end

    @eval function Base.$fn(d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(d)
        p1_out, p2_out = Base.$fn(x)
        t1_out, t2_out = gradient_tracer_1_to_2(
            t, is_firstder_out1_zero_local($fn, x), is_firstder_out2_zero_local($fn, x)
        )
        return (Dual(p1_out, t1_out), Dual(p1_out, t1_out))
    end
end

# TODO: support Dual tracers for these.
# Extra types required for exponent 
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::GradientTracer, ::$T) = t
    @eval Base.:^(::$T, t::GradientTracer) = t
end
Base.:^(t::GradientTracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::GradientTracer) = t

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GradientTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:GradientTracer} = empty(T)
