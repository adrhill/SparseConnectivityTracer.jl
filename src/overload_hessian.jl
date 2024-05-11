# REVIEW TODO: this keeps popping up so often that think it's worth an abstraction  
# This was previously called "promote_order".
# The overly descriptive name is TEMPORARY.
function tracer_from_hessian_OR_outer_product_OR_of_gradient(
    t::T
) where {T<:GlobalHessianTracer}
    hessian_out = deepcopy(t.hessian)
    hessian_out = outer_product_or!(hessian_out, t.gradient, t.gradient)
    return T(t.gradient, hessian_out)
end

# ## 1-to-1
for fn in ops_1_to_1
    @eval function Base.$fn(t::T) where {T<:GlobalHessianTracer}
        if is_seconder_zero_global($fn)
            if is_firstder_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
        end
    end
end

# ## 2-to-1
for fn in ops_2_to_1
    @eval function Base.$fn(a::T, b::T) where {G,H,T<:GlobalHessianTracer{G,H}}
        gradient_out, hessian_out = if !is_firstder_arg1_zero_global($fn)
            if !is_firstder_arg2_zero_global($fn)
                (a.gradient ∨ b.gradient, a.hessian ∨ b.hessian)
            else
                (a.gradient, deepcopy(a.hessian))
            end
        else # is_firstder_arg1_zero_global
            if !is_firstder_arg2_zero_global($fn)
                (b.gradient, deepcopy(b.hessian))
            else
                (empty_sparse_vector(G), empty_sparse_matrix(H))
            end
        end

        if !is_seconder_arg1_zero_global($fn)
            outer_product_or!(hessian_out, a.gradient, a.gradient)
        end
        if !is_seconder_arg2_zero_global($fn)
            outer_product_or!(hessian_out, b.gradient, b.gradient)
        end
        if !is_crossder_zero_global($fn)
            outer_product_or!(hessian_out, a.gradient, b.gradient)
            outer_product_or!(hessian_out, b.gradient, a.gradient)
        end
        return T(gradient_out, hessian_out)
    end

    @eval function Base.$fn(t::T, ::Number) where {T<:GlobalHessianTracer}
        if is_seconder_arg1_zero_global($fn)
            if is_firstder_arg1_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
        end
    end
    @eval function Base.$fn(::Number, t::T) where {T<:GlobalHessianTracer}
        if is_seconder_arg2_zero_global($fn)
            if is_firstder_arg2_zero_global($fn)
                return empty(T)
            else
                return t
            end
        else
            return tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
        end
    end
end

# Extra types required for exponent
for T in (:Real, :Integer, :Rational)
    @eval Base.:^(t::GlobalHessianTracer, ::$T) =
        tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
    @eval Base.:^(::$T, t::GlobalHessianTracer) =
        tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
end
function Base.:^(t::GlobalHessianTracer, ::Irrational{:ℯ})
    return tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
end
function Base.:^(::Irrational{:ℯ}, t::GlobalHessianTracer)
    return tracer_from_hessian_OR_outer_product_OR_of_gradient(t)
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GlobalHessianTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:GlobalHessianTracer} = empty(T)
