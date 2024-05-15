## 1-to-1
for fn in ops_1_to_1
    @eval function Base.$fn(t::T) where {T<:GradientTracer}
        if is_firstder_zero_global($fn)
            return empty(T)
        else
            return t
        end
    end
    @eval function Base.$fn(t::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(t)
        out = Base.$fn(x)
        if is_firstder_zero_local($fn, x)
            return Dual(out, empty(T))
        else
            return Dual(out, t)
        end
    end
end

## 2-to-1
for fn in ops_2_to_1
    @eval function Base.$fn(tx::T, ty::T) where {T<:GradientTracer}
        ∂f∂x0 = is_firstder_arg1_zero_global($fn)
        ∂f∂y0 = is_firstder_arg2_zero_global($fn)
        if ∂f∂x0
            if ∂f∂y0
                return empty(T)
            else # ∂f∂y ≠ 0 
                return ty
            end
        else # ∂f∂x ≠ 0 
            if ∂f∂y0
                return tx
            else # ∂f∂y ≠ 0 
                return T(gradient(tx) ∪ gradient(ty))
            end
        end
    end
    @eval function Base.$fn(tx::D, ty::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(tx)
        y = primal(ty)
        out = Base.$fn(x, y)

        ∂f∂x0 = is_firstder_arg1_zero_local($fn, x, y)
        ∂f∂y0 = is_firstder_arg2_zero_local($fn, x, y)
        if ∂f∂x0
            if ∂f∂y0
                return Dual(out, empty(T))
            else # ∂f∂y ≠ 0 
                return Dual(out, ty)
            end
        else # ∂f∂x ≠ 0 
            if ∂f∂y0
                return Dual(out, tx)
            else # ∂f∂y ≠ 0 
                return Dual(out, T(gradient(tx) ∪ gradient(ty)))
            end
        end
    end

    @eval function Base.$fn(t::T, ::Number) where {T<:GradientTracer}
        if is_firstder_arg1_zero_global($fn)
            return empty(T)
        else
            return t
        end
    end
    @eval function Base.$fn(tx::D, y::Number) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(tx)
        out = Base.$fn(x, y)
        if is_firstder_arg1_zero_local($fn, x, y)
            return Dual(out, empty(T))
        else
            return Dual(out, t)
        end
    end

    @eval function Base.$fn(::Number, t::T) where {T<:GradientTracer}
        if is_firstder_arg2_zero_global($fn)
            return empty(T)
        else
            return t
        end
    end
    @eval function Base.$fn(x::Number, ty::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        y = primal(ty)
        out = Base.$fn(x, y)
        if is_firstder_arg2_zero_local($fn, x, y)
            return Dual(out, empty(T))
        else
            return Dual(out, t)
        end
    end
end

## 1-to-2
for fn in ops_1_to_2
    @eval function Base.$fn(t::T) where {T<:GradientTracer}
        tracer1 = if is_firstder_out1_zero_global($fn)
            empty(T)
        else
            t
        end
        tracer2 = if is_firstder_out2_zero_global($fn)
            empty(T)
        else
            t
        end
        return (tracer1,)
    end

    @eval function Base.$fn(tx::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(tx)
        out1, out2 = Base.$fn(x)

        tracer1 = if is_firstder_out1_zero_global($fn)
            empty(T)
        else
            t
        end
        tracer2 = if is_firstder_out2_zero_global($fn)
            empty(T)
        else
            t
        end
        return (Dual(out1, tracer1), Dual(out2, tracer2))
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
