## 1-to-1

@noinline function gradient_tracer_1_to_1(t::T, is_firstder_zero::Bool) where {T<:GradientTracer}
    if t.isempty # TODO: add test
        return t
    else
        gradient = gradient_tracer_1_to_1_inner(t.gradient, is_firstder_zero)
        return T(gradient) # return tracer
    end
end

function gradient_tracer_1_to_1_inner(s::S, is_firstder_zero::Bool) where {S<:AbstractSet}
    if is_firstder_zero
        return myempty(S)
    else
        return s # return set
    end
end

function overload_gradient_1_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_1(t, $SCT.is_firstder_zero_global($M.$op))
        end
    end
end

function overload_gradient_1_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            x = d.primal
            p_out = $M.$op(x)
            t_out = $SCT.gradient_tracer_1_to_1(
                d.tracer, $SCT.is_firstder_zero_local($op, x)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

@noinline function gradient_tracer_2_to_1(
    tx::T, ty::T, is_firstder_arg1_zero::Bool, is_firstder_arg2_zero::Bool
) where {T<:GradientTracer}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return gradient_tracer_1_to_1(tx, is_firstder_arg1_zero)
    elseif tx.isempty
        return gradient_tracer_1_to_1(ty, is_firstder_arg2_zero)
    else
        gradient = gradient_tracer_2_to_1_inner(
            tx.gradient, ty.gradient, is_firstder_arg1_zero, is_firstder_arg2_zero
        )
        return T(gradient) # return tracer
    end
end

function gradient_tracer_2_to_1_inner(
    sx::S, sy::S, is_firstder_arg1_zero::Bool, is_firstder_arg2_zero::Bool
) where {S<:AbstractSet}
    if is_firstder_arg1_zero && is_firstder_arg2_zero
        return myempty(S)
    elseif !is_firstder_arg1_zero && is_firstder_arg2_zero
        return sx
    elseif is_firstder_arg1_zero && !is_firstder_arg2_zero
        return sy
    else
        return union(sx, sy) # return set
    end
end

function overload_gradient_2_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.GradientTracer}
            return $SCT.gradient_tracer_2_to_1(
                tx,
                ty,
                $SCT.is_firstder_arg1_zero_global($M.$op),
                $SCT.is_firstder_arg2_zero_global($M.$op),
            )
        end

        function $M.$op(tx::$SCT.GradientTracer, ::Real)
            return $SCT.gradient_tracer_1_to_1(
                tx, $SCT.is_firstder_arg1_zero_global($M.$op)
            )
        end

        function $M.$op(::Real, ty::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_1(
                ty, $SCT.is_firstder_arg2_zero_global($M.$op)
            )
        end
    end
end

function overload_gradient_2_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(dx::D, dy::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            x = dx.primal
            y = dy.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_2_to_1(
                dx.tracer,
                dy.tracer,
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(dx::D, y::Real) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            x = dx.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_1_to_1(
                dx.tracer, $SCT.is_firstder_arg1_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(x::Real, dy::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            y = dy.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_1_to_1(
                dy.tracer, $SCT.is_firstder_arg2_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

@noinline function gradient_tracer_1_to_2(
    t::T, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {T<:GradientTracer}
    if t.isempty # TODO: add test
        return (t, t)
    else
        gradient1, gradient2 = gradient_tracer_1_to_2_inner(
            t.gradient, is_firstder_out1_zero, is_firstder_out2_zero
        )
        return (T(gradient1), T(gradient2)) # return tracers
    end
end

function gradient_tracer_1_to_2_inner(
    s::S, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {S<:AbstractSet}
    set1 = gradient_tracer_1_to_1_inner(s, is_firstder_out1_zero)
    set2 = gradient_tracer_1_to_1_inner(s, is_firstder_out2_zero)
    return (set1, set2)
end

function overload_gradient_1_to_2(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.GradientTracer)
            return $SCT.gradient_tracer_1_to_2(
                t,
                $SCT.is_firstder_out1_zero_global($M.$op),
                $SCT.is_firstder_out2_zero_global($M.$op),
            )
        end
    end
end

function overload_gradient_1_to_2_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            x = d.primal
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.gradient_tracer_1_to_2(
                d.tracer,
                $SCT.is_firstder_out1_zero_local($M.$op, x),
                $SCT.is_firstder_out2_zero_local($M.$op, x),
            )
            return ($SCT.Dual(p1_out, t1_out), $SCT.Dual(p2_out, t2_out))  # TODO: this was wrong, add test
        end
    end
end

## Special cases

## Exponent (requires extra types)
for S in (Integer, Rational, Irrational{:ℯ})
    function Base.:^(t::T, ::S) where {T<:GradientTracer}
        gradient = gradient_tracer_1_to_1_inner(t.gradient, false)
        return T(gradient)
    end
    function Base.:^(::S, t::T) where {T<:GradientTracer}
        gradient = gradient_tracer_1_to_1_inner(t.gradient, false)
        return T(gradient)
    end
    function Base.:^(d::D, y::S) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = d.primal
        t = d.tracer
        gradient = gradient_tracer_1_to_1_inner(t.gradient, false)
        return Dual(x^y, T(gradient))
    end
    function Base.:^(x::S, d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        y = d.primal
        t = d.tracer
        gradient = gradient_tracer_1_to_1_inner(t.gradient, false)
        return Dual(x^y, T(gradient))
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GradientTracer} = myempty(T)
function Base.round(
    d::D, mode::RoundingMode; kwargs...
) where {P,T<:GradientTracer,D<:Dual{P,T}}
    return Dual(round(d.primal, mode; kwargs...), myempty(T))
end

## Random numbers 
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:GradientTracer} = myempty(T)  # TODO: was missing Base, add tests
