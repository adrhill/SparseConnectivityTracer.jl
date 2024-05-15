## 1-to-1
for fn in ops_1_to_1
    @eval function Base.$fn(t::T) where {T<:HessianTracer}
        if is_seconder_zero_global($fn)
            if is_firstder_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
    end
    @eval function Base.$fn(t::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(t)
        out = Base.$fn(x)
        if is_seconder_zero_local($fn, x)
            if is_firstder_zero_local($fn, x)
                return Dual(out, empty(T))
            else
                return Dual(out, t)
            end
        else
            return Dual(out, T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t))))
        end
    end
end

## 2-to-1
for fn in ops_2_to_1
    @eval function Base.$fn(a::T, b::T) where {G,H,T<:HessianTracer{G,H}}
        grad = empty(G)
        hess = empty(H)
        if !is_firstder_arg1_zero_global($fn)
            grad = union(grad, gradient(a)) # TODO: use union!
            union!(hess, hessian(a))
        end
        if !is_firstder_arg2_zero_global($fn)
            grad = union(grad, gradient(b)) # TODO: use union!
            union!(hess, hessian(b))
        end
        if !is_seconder_arg1_zero_global($fn)
            union!(hess, gradient(a) × gradient(a))
        end
        if !is_seconder_arg2_zero_global($fn)
            union!(hess, gradient(b) × gradient(b))
        end
        if !is_crossder_zero_global($fn)
            union!(hess, (gradient(a) × gradient(b)) ∪ (gradient(b) × gradient(a)))
        end
        return T(grad, hess)
    end
    @eval function Base.$fn(a::D, b::D) where {P,G,H,T<:HessianTracer{G,H},D<:Dual{P,T}}
        x = primal(a)
        y = primal(b)
        out = Base.$fn(x, y)

        grad = empty(G)
        hess = empty(H)
        if !is_firstder_arg1_zero_local($fn, x, y)
            grad = union(grad, gradient(a)) # TODO: use union!
            union!(hess, hessian(a))
        end
        if !is_firstder_arg2_zero_local($fn, x, y)
            grad = union(grad, gradient(b)) # TODO: use union!
            union!(hess, hessian(b))
        end
        if !is_seconder_arg1_zero_local($fn, x, y)
            union!(hess, gradient(a) × gradient(a))
        end
        if !is_seconder_arg2_zero_local($fn, x, y)
            union!(hess, gradient(b) × gradient(b))
        end
        if !is_crossder_zero_local($fn, x, y)
            union!(hess, (gradient(a) × gradient(b)) ∪ (gradient(b) × gradient(a)))
        end
        return Dual(out, T(grad, hess))
    end

    @eval function Base.$fn(t::T, ::Number) where {G,H,T<:HessianTracer{G,H}}
        if is_seconder_arg1_zero_global($fn)
            if is_firstder_arg1_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
    end
    @eval function Base.$fn(
        t::D, y::Number
    ) where {P,G,H,T<:HessianTracer{G,H},D<:Dual{P,T}}
        x = primal(t)
        out = Base.$fn(x, y)
        if is_seconder_arg1_zero_local($fn, x, y)
            if is_firstder_arg1_zero_local($fn, x, y)
                return Dual(out, empty(T))
            else
                return Dual(out, t)
            end
        else
            return Dual(out, T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t))))
        end
    end

    @eval function Base.$fn(x::Number, t::T) where {G,H,T<:HessianTracer{G,H}}
        if is_seconder_arg2_zero_global($fn)
            if is_firstder_arg2_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
    end
    @eval function Base.$fn(
        x::Number, t::D
    ) where {P,G,H,T<:HessianTracer{G,H},D<:Dual{P,T}}
        y = primal(t)
        out = Base.$fn(x, y)
        if is_seconder_arg2_zero_local($fn, x, y)
            if is_firstder_arg2_zero_local($fn, x, y)
                return Dual(out, empty(T))
            else
                return Dual(out, t)
            end
        else
            return Dual(out, T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t))))
        end
    end
end

## 1-to-2
for fn in ops_1_to_2
    @eval function Base.$fn(t::T) where {T<:HessianTracer}
        tracer1 = if is_seconder_out1_zero_global($fn)
            if is_firstder_out1_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
        tracer2 = if is_seconder_out2_zero_global($fn)
            if is_firstder_out2_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
        return (tracer1, tracer2)
    end

    @eval function Base.$fn(tx::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(tx)
        out1, out2 = Base.$fn(x)

        tracer1 = if is_seconder_out1_zero_local($fn, x)
            if is_firstder_out1_zero_local($fn, x)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
        tracer2 = if is_seconder_out2_zero_local($fn, x)
            if is_firstder_out2_zero_local($fn, x)
                return empty(T)
            else
                return t
            end
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
        return (Dual(out1, tracer1), Dual(out2, tracer2))
    end
end

# TODO: support Dual tracers for these.
# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval function Base.:^(t::T, ::$T) where {T<:HessianTracer}
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
    @eval function Base.:^(::$T, t::T) where {T<:HessianTracer}
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
end
function Base.:^(t::T, ::Irrational{:ℯ}) where {T<:HessianTracer}
    return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
end
function Base.:^(::Irrational{:ℯ}, t::T) where {T<:HessianTracer}
    return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:HessianTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = empty(T)
