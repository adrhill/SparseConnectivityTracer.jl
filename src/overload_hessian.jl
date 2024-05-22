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

function overload_hessian_1_to_1(M, op)
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(
                t,
                $SCT.is_firstder_zero_global($M.$op),
                $SCT.is_seconder_zero_global($M.$op),
            )
        end
        function $M.$op(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out = $M.$op(x)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(d),
                $SCT.is_firstder_zero_local($M.$op, x),
                $SCT.is_seconder_zero_local($M.$op, x),
            )
            return $SCT.Dual(p_out, t_out)
        end
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

function overload_hessian_2_to_1(M, op)
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.HessianTracer}
            return $SCT.hessian_tracer_2_to_1(
                tx,
                ty,
                $SCT.is_firstder_arg1_zero_global($M.$op),
                $SCT.is_seconder_arg1_zero_global($M.$op),
                $SCT.is_firstder_arg2_zero_global($M.$op),
                $SCT.is_seconder_arg2_zero_global($M.$op),
                $SCT.is_crossder_zero_global($M.$op),
            )
        end
        function $M.$op(dx::D, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_2_to_1(
                $SCT.tracer(dx),
                $SCT.tracer(dy),
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg1_zero_local($M.$op, x, y),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg2_zero_local($M.$op, x, y),
                $SCT.is_crossder_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(tx::$SCT.HessianTracer, y::Number)
            return $SCT.hessian_tracer_1_to_1(
                tx,
                $SCT.is_firstder_arg1_zero_global($M.$op),
                $SCT.is_seconder_arg1_zero_global($M.$op),
            )
        end
        function $M.$op(x::Number, ty::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_1(
                ty,
                $SCT.is_firstder_arg2_zero_global($M.$op),
                $SCT.is_seconder_arg2_zero_global($M.$op),
            )
        end

        function $M.$op(dx::D, y::Number) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(dx),
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg1_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end
        function $M.$op(x::Number, dy::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.hessian_tracer_1_to_1(
                $SCT.tracer(dy),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
                $SCT.is_seconder_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end
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

function overload_hessian_1_to_2(M, op)
    return quote
        function $M.$op(t::$SCT.HessianTracer)
            return $SCT.hessian_tracer_1_to_2(
                t,
                $SCT.is_firstder_out1_zero_global($M.$op),
                $SCT.is_seconder_out1_zero_global($M.$op),
                $SCT.is_firstder_out2_zero_global($M.$op),
                $SCT.is_seconder_out2_zero_global($M.$op),
            )
        end

        function $M.$op(d::D) where {P,T<:$SCT.HessianTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.hessian_tracer_1_to_2(
                d,
                $SCT.is_firstder_out1_zero_local($M.$op, x),
                $SCT.is_seconder_out1_zero_local($M.$op, x),
                $SCT.is_firstder_out2_zero_local($M.$op, x),
                $SCT.is_seconder_out2_zero_local($M.$op, x),
            )
            return ($SCT.Dual(p1_out, t1_out), $SCT.Dual(p2_out, t2_out))
        end
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
