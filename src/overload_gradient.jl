## 1-to-1

function gradient_tracer_1_to_1(t::T, is_firstder_zero::Bool) where {T<:GradientTracer}
    s = gradient(t)
    s_out = gradient_tracer_1_to_1(s, is_firstder_zero)
    return T(s_out)
end

function gradient_tracer_1_to_1(
    s::S, is_firstder_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_firstder_zero
        return myempty(S)
    else
        return s
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
            x = $SCT.primal(d)
            p_out = $M.$op(x)
            t_out = $SCT.gradient_tracer_1_to_1(
                $SCT.tracer(d), $SCT.is_firstder_zero_local($op, x)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

function gradient_tracer_2_to_1(
    tx::T, ty::T, is_firstder_arg1_zero::Bool, is_firstder_arg2_zero::Bool
) where {T<:GradientTracer}
    sx, sy = gradient(tx), gradient(ty)
    s_out = gradient_tracer_2_to_1(sx, sy, is_firstder_arg1_zero, is_firstder_arg2_zero)
    return T(s_out)
end

function gradient_tracer_2_to_1(
    sx::S, sy::S, is_firstder_arg1_zero::Bool, is_firstder_arg2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_firstder_arg1_zero && is_firstder_arg2_zero
        return myempty(S)
    elseif !is_firstder_arg1_zero && is_firstder_arg2_zero
        return sx
    elseif is_firstder_arg1_zero && !is_firstder_arg2_zero
        return sy
    else
        return union(sx, sy)
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
            x = $SCT.primal(dx)
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_2_to_1(
                $SCT.tracer(dx),
                $SCT.tracer(dy),
                $SCT.is_firstder_arg1_zero_local($M.$op, x, y),
                $SCT.is_firstder_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(dx::D, y::Real) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_1_to_1(
                $SCT.tracer(dx), $SCT.is_firstder_arg1_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(x::Real, dy::D) where {P,T<:$SCT.GradientTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.gradient_tracer_1_to_1(
                $SCT.tracer(dy), $SCT.is_firstder_arg2_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

function gradient_tracer_1_to_2(
    t::T, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {T<:GradientTracer}
    s = gradient(t)
    s_out1, s_out2 = gradient_tracer_1_to_2(s, is_firstder_out1_zero, is_firstder_out2_zero)
    return (T(s_out1), T(s_out2))
end

function gradient_tracer_1_to_2(
    s::S, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    s_out1 = gradient_tracer_1_to_1(s, is_firstder_out1_zero)
    s_out2 = gradient_tracer_1_to_1(s, is_firstder_out2_zero)
    return (s_out1, s_out2)
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
            x = $SCT.primal(d)
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.gradient_tracer_1_to_2(
                $SCT.tracer(d),
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
    Base.:^(t::GradientTracer, ::S) = t
    Base.:^(::S, t::GradientTracer) = t

    function Base.:^(dx::D, y::S) where {P,T<:GradientTracer,D<:Dual{P,T}}
        return Dual(primal(dx)^y, tracer(dx))
    end
    function Base.:^(x::S, dy::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        return Dual(x^primal(dy), tracer(dy))
    end
end

## Rounding
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GradientTracer} = myempty(T)
function Base.round(
    d::D, mode::RoundingMode; kwargs...
) where {P,T<:GradientTracer,D<:Dual{P,T}}
    return Dual(round(primal(d), mode; kwargs...), myempty(T))
end

## Random numbers 
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:GradientTracer} = myempty(T)  # TODO: was missing Base, add tests
