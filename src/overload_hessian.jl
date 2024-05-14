## 1-to-1
for fn in ops_1_to_1
    @eval function Base.$fn(t::T) where {T<:GlobalHessianTracer}
        if is_seconder_zero_global($fn)
            if is_firstder_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(t.grad, t.hess ∪ (t.grad × t.grad))
        end
    end
end

## 2-to-1
for fn in ops_2_to_1
    @eval function Base.$fn(a::T, b::T) where {G,H,T<:GlobalHessianTracer{G,H}}
        grad = empty(G)
        hess = empty(H)
        if !is_firstder_arg1_zero_global($fn)
            grad = union(grad, a.grad) # TODO: use union!
            union!(hess, a.hess)
        end
        if !is_firstder_arg2_zero_global($fn)
            grad = union(grad, b.grad) # TODO: use union!
            union!(hess, b.hess)
        end
        if !is_seconder_arg1_zero_global($fn)
            union!(hess, a.grad × a.grad)
        end
        if !is_seconder_arg2_zero_global($fn)
            union!(hess, b.grad × b.grad)
        end
        if !is_crossder_zero_global($fn)
            union!(hess, (a.grad × b.grad) ∪ (b.grad × a.grad))
        end
        return T(grad, hess)
    end

    @eval function Base.$fn(t::T, ::Number) where {G,H,T<:GlobalHessianTracer{G,H}}
        if is_seconder_arg1_zero_global($fn)
            if is_firstder_arg1_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(t.grad, t.hess ∪ (t.grad × t.grad))
        end
    end
    @eval function Base.$fn(::Number, t::T) where {G,H,T<:GlobalHessianTracer{G,H}}
        if is_seconder_arg2_zero_global($fn)
            if is_firstder_arg2_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return T(t.grad, t.hess ∪ (t.grad × t.grad))
        end
    end
end

# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval function Base.:^(t::T, ::$T) where {T<:GlobalHessianTracer}
        return T(t.grad, t.hess ∪ (t.grad × t.grad))
    end
    @eval function Base.:^(::$T, t::T) where {T<:GlobalHessianTracer}
        return T(t.grad, t.hess ∪ (t.grad × t.grad))
    end
end
function Base.:^(t::T, ::Irrational{:ℯ}) where {T<:GlobalHessianTracer}
    return T(t.grad, t.hess ∪ (t.grad × t.grad))
end
function Base.:^(::Irrational{:ℯ}, t::T) where {T<:GlobalHessianTracer}
    return T(t.grad, t.hess ∪ (t.grad × t.grad))
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GlobalHessianTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:GlobalHessianTracer} = empty(T)
