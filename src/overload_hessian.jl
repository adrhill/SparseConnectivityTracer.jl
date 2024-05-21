## 1-to-1

function hessian_tracer_1_to_1(
    t::T, is_firstder_zero::Bool, is_seconder_zero::Bool
) where {G,H,T<:HessianTracer{G,H}}
    if is_seconder_zero
        if is_firstder_zero
            return empty(T)
        else
            return t
        end
    else
        if is_firstder_zero
            return T(empty(G), gradient(t) × gradient(t))
        else
            return T(gradient(t), hessian(t) ∪ (gradient(t) × gradient(t)))
        end
    end
end

function overload_hessian_1_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(t::HessianTracer)
        return hessian_tracer_1_to_1(
            t, is_firstder_zero_global($ms.$fns), is_seconder_zero_global($ms.$fns)
        )
    end
    @eval function $ms.$fns(d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        p_out = $ms.$fns(x)
        t_out = hessian_tracer_1_to_1(
            tracer(d),
            is_firstder_zero_local($ms.$fns, x),
            is_seconder_zero_local($ms.$fns, x),
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

function overload_hessian_2_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(tx::T, ty::T) where {T<:HessianTracer}
        return hessian_tracer_2_to_1(
            tx,
            ty,
            is_firstder_arg1_zero_global($ms.$fns),
            is_seconder_arg1_zero_global($ms.$fns),
            is_firstder_arg2_zero_global($ms.$fns),
            is_seconder_arg2_zero_global($ms.$fns),
            is_crossder_zero_global($ms.$fns),
        )
    end
    @eval function $ms.$fns(dx::D, dy::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(dx)
        y = primal(dy)
        p_out = $ms.$fns(x, y)
        t_out = hessian_tracer_2_to_1(
            tracer(dx),
            tracer(dy),
            is_firstder_arg1_zero_local($ms.$fns, x, y),
            is_seconder_arg1_zero_local($ms.$fns, x, y),
            is_firstder_arg2_zero_local($ms.$fns, x, y),
            is_seconder_arg2_zero_local($ms.$fns, x, y),
            is_crossder_zero_local($ms.$fns, x, y),
        )
        return Dual(p_out, t_out)
    end

    @eval function $ms.$fns(tx::HessianTracer, y::Number)
        return hessian_tracer_1_to_1(
            tx,
            is_firstder_arg1_zero_global($ms.$fns),
            is_seconder_arg1_zero_global($ms.$fns),
        )
    end
    @eval function $ms.$fns(x::Number, ty::HessianTracer)
        return hessian_tracer_1_to_1(
            ty,
            is_firstder_arg2_zero_global($ms.$fns),
            is_seconder_arg2_zero_global($ms.$fns),
        )
    end

    @eval function $ms.$fns(dx::D, y::Number) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(dx)
        p_out = $ms.$fns(x, y)
        t_out = hessian_tracer_1_to_1(
            tracer(dx),
            is_firstder_arg1_zero_local($ms.$fns, x, y),
            is_seconder_arg1_zero_local($ms.$fns, x, y),
        )
        return Dual(p_out, t_out)
    end
    @eval function $ms.$fns(x::Number, dy::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        y = primal(dy)
        p_out = $ms.$fns(x, y)
        t_out = hessian_tracer_1_to_1(
            tracer(dy),
            is_firstder_arg2_zero_local($ms.$fns, x, y),
            is_seconder_arg2_zero_local($ms.$fns, x, y),
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

function overload_hessian_1_to_2(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(t::HessianTracer)
        return hessian_tracer_1_to_2(
            t,
            is_firstder_out1_zero_global($ms.$fns),
            is_seconder_out1_zero_global($ms.$fns),
            is_firstder_out2_zero_global($ms.$fns),
            is_seconder_out2_zero_global($ms.$fns),
        )
    end

    @eval function $ms.$fns(d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        p1_out, p2_out = $ms.$fns(x)
        t1_out, t2_out = hessian_tracer_1_to_2(
            d,
            is_firstder_out1_zero_local($ms.$fns, x),
            is_seconder_out1_zero_local($ms.$fns, x),
            is_firstder_out2_zero_local($ms.$fns, x),
            is_seconder_out2_zero_local($ms.$fns, x),
        )
        return (Dual(p1_out, t1_out), Dual(p2_out, t2_out))
    end
end

## Special cases

## Exponent (requires extra types)
# TODO: support Dual tracers for these.

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
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:HessianTracer} = empty(T)  # TODO: was missing Base, add tests
