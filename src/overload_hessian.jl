## 1-to-1
function hessian_tracer_1_to_1(
    t::T, is_firstder_zero::Bool, is_seconder_zero::Bool
) where {T<:HessianTracer}
    if is_seconder_zero
        if is_firstder_zero
            return empty(T)
        else
            return t
        end
    else
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
end

for fn in ops_1_to_1
    @eval function Base.$fn(t::HessianTracer)
        return hessian_tracer_1_to_1(
            t, is_firstder_zero_global($fn), is_seconder_zero_global($fn)
        )
    end
    @eval function Base.$fn(d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        p_out = Base.$fn(x)
        t_out = hessian_tracer_1_to_1(
            tracer(d), is_firstder_zero_local($fn, x), is_seconder_zero_local($fn, x)
        )
        return Dual(p_out, t_out)
    end
end

## 2-to-1
function hessian_tracer_2_to_1(
    a::T,
    b::T,
    is_firstder_arg1_zero::Bool,
    is_seconder_arg1_zero::Bool,
    is_firstder_arg2_zero::Bool,
    is_seconder_arg2_zero::Bool,
    is_crossder_zero::Bool,
) where {G,H,T<:HessianTracer{G,H}}
    grad = empty(G)
    hess = empty(H)
    if !is_firstder_arg1_zero
        grad = union(grad, gradient(a)) # TODO: use union!
        union!(hess, hessian(a))
    end
    if !is_firstder_arg2_zero
        grad = union(grad, gradient(b)) # TODO: use union!
        union!(hess, hessian(b))
    end
    if !is_seconder_arg1_zero
        union!(hess, gradient(a) × gradient(a))
    end
    if !is_seconder_arg2_zero
        union!(hess, gradient(b) × gradient(b))
    end
    if !is_crossder_zero
        union!(hess, (gradient(a) × gradient(b)) ∪ (gradient(b) × gradient(a)))
    end
    return T(grad, hess)
end

function hessian_tracer_2_to_1_one_tracer(
    t::T, is_firstder_zero::Bool, is_seconder_zero::Bool
) where {T<:HessianTracer}
    # NOTE: this is identical to hessian_tracer_1_to_1 due to ignored second argument having empty set
    # TODO: remove once gdalle agrees
    if is_seconder_zero
        if is_firstder_zero
            return empty(T)
        else
            return t
        end
    else
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
end

for fn in ops_2_to_1
    @eval function Base.$fn(tx::T, ty::T) where {T<:HessianTracer}
        return hessian_tracer_2_to_1(
            tx,
            ty,
            is_firstder_arg1_zero_global($fn),
            is_seconder_arg1_zero_global($fn),
            is_firstder_arg2_zero_global($fn),
            is_seconder_arg2_zero_global($fn),
            is_crossder_zero_global($fn),
        )
    end
    @eval function Base.$fn(dx::D, dy::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(dx)
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = hessian_tracer_2_to_1(
            tracer(dx),
            tracer(dy),
            is_firstder_arg1_zero_local($fn, x, y),
            is_seconder_arg1_zero_local($fn, x, y),
            is_firstder_arg2_zero_local($fn, x, y),
            is_seconder_arg2_zero_local($fn, x, y),
            is_crossder_zero_local($fn, x, y),
        )
        return Dual(p_out, t_out)
    end

    @eval function Base.$fn(tx::HessianTracer, y::Number)
        return hessian_tracer_2_to_1_one_tracer(
            tx, is_firstder_arg1_zero_global($fn), is_seconder_arg1_zero_global($fn)
        )
    end
    @eval function Base.$fn(x::Number, ty::HessianTracer)
        return hessian_tracer_2_to_1_one_tracer(
            ty, is_firstder_arg2_zero_global($fn), is_seconder_arg2_zero_global($fn)
        )
    end

    @eval function Base.$fn(dx::D, y::Number) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(dx)
        p_out = Base.$fn(x, y)
        t_out = hessian_tracer_2_to_1_one_tracer(
            tracer(dx),
            is_firstder_arg1_zero_local($fn, x, y),
            is_seconder_arg1_zero_local($fn, x, y),
        )
        return Dual(p_out, t_out)
    end
    @eval function Base.$fn(x::Number, dy::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = hessian_tracer_2_to_1_one_tracer(
            tracer(dy),
            is_firstder_arg2_zero_local($fn, x, y),
            is_seconder_arg2_zero_local($fn, x, y),
        )
        return Dual(p_out, t_out)
    end
end

## 1-to-2
function hessian_tracer_1_to_2(
    t::T,
    is_firstder_out1_zero::Bool,
    is_seconder_out1_zero::Bool,
    is_firstder_out2_zero::Bool,
    is_seconder_out2_zero::Bool,
) where {T<:HessianTracer}
    t1 = hessian_tracer_1_to_1(t, is_firstder_out1_zero, is_seconder_out1_zero)
    t2 = hessian_tracer_1_to_1(t, is_firstder_out2_zero, is_seconder_out2_zero)
    return (t1, t2)
end
for fn in ops_1_to_2
    @eval function Base.$fn(t::HessianTracer)
        return hessian_tracer_1_to_2(
            t,
            is_firstder_out1_zero_global($fn),
            is_seconder_out1_zero_global($fn),
            is_firstder_out2_zero_global($fn),
            is_seconder_out2_zero_global($fn),
        )
    end

    @eval function Base.$fn(d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        p1_out, p2_out = Base.$fn(x)
        t1_out, t2_out = hessian_tracer_1_to_2(
            d,
            is_firstder_out1_zero_local($fn, x),
            is_seconder_out1_zero_local($fn, x),
            is_firstder_out2_zero_local($fn, x),
            is_seconder_out2_zero_local($fn, x),
        )
        return (Dual(p1_out, t1_out), Dual(p2_out, t2_out))
    end
end

# TODO: support Dual tracers for these.
# Extra types required for exponent
for S in (Real, Integer, Rational, Irrational{:ℯ})
    function Base.:^(t::T, ::S) where {T<:HessianTracer}
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
    function Base.:^(::S, t::T) where {T<:HessianTracer}
        return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:HessianTracer} = empty(T)

## Random numbers
rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = empty(T)
