## 1-to-1

function connectivity_tracer_1_to_1(
    t::T, is_influence_zero::Bool
) where {T<:ConnectivityTracer}
    s = inputs(t)
    s_out = connectivity_tracer_1_to_1(s, is_influence_zero)
    return T(s_out)
end

function connectivity_tracer_1_to_1(
    s::S, is_influence_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_influence_zero
        return myempty(S)
    else
        return s
    end
end

function overload_connectivity_1_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::T) where {T<:$SCT.ConnectivityTracer}
            return $SCT.connectivity_tracer_1_to_1(t, $SCT.is_influence_zero_global($M.$op))
        end
    end
end

function overload_connectivity_1_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out = $M.$op(x)
            t_out = $SCT.connectivity_tracer_1_to_1(
                $SCT.tracer(d), $SCT.is_influence_zero_local($M.$op, x)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

function connectivity_tracer_2_to_1(
    tx::T, ty::T, is_influence_arg1_zero::Bool, is_influence_arg2_zero::Bool
) where {T<:ConnectivityTracer}
    sx, sy = inputs(tx), inputs(ty)
    s_out = connectivity_tracer_2_to_1(
        sx, sy, is_influence_arg1_zero, is_influence_arg2_zero
    )
    return T(s_out)
end

function connectivity_tracer_2_to_1(
    sx::S, sy::S, is_influence_arg1_zero::Bool, is_influence_arg2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_influence_arg1_zero && is_influence_arg2_zero
        return myempty(S)
    elseif !is_influence_arg1_zero && is_influence_arg2_zero
        return sx
    elseif is_influence_arg1_zero && !is_influence_arg2_zero
        return sy
    else
        return clever_union(sx, sy)
    end
end

function overload_connectivity_2_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.ConnectivityTracer}
            return $SCT.connectivity_tracer_2_to_1(
                tx,
                ty,
                $SCT.is_influence_arg1_zero_global($M.$op),
                $SCT.is_influence_arg2_zero_global($M.$op),
            )
        end

        function $M.$op(tx::$SCT.ConnectivityTracer, ::Real)
            return $SCT.connectivity_tracer_1_to_1(
                tx, $SCT.is_influence_arg1_zero_global($M.$op)
            )
        end

        function $M.$op(::Real, ty::$SCT.ConnectivityTracer)
            return $SCT.connectivity_tracer_1_to_1(
                ty, $SCT.is_influence_arg2_zero_global($M.$op)
            )
        end
    end
end

function overload_connectivity_2_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(dx::D, dy::D) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_2_to_1(
                $SCT.tracer(dx),
                $SCT.tracer(dy),
                $SCT.is_influence_arg1_zero_local($M.$op, x, y),
                $SCT.is_influence_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            dx::D, y::Real
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(
                $SCT.tracer(dx), $SCT.is_influence_arg1_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            x::Real, dy::D
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(
                $SCT.tracer(dy), $SCT.is_influence_arg2_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

function connectivity_tracer_1_to_2(
    t::T, is_influence_out1_zero::Bool, is_influence_out2_zero::Bool
) where {T<:ConnectivityTracer}
    s = inputs(t)
    (s_out1, s_out2) = connectivity_tracer_1_to_2(
        s, is_influence_out1_zero, is_influence_out2_zero
    )
    return (T(s_out1), T(s_out2))
end

function connectivity_tracer_1_to_2(
    s::S, is_influence_out1_zero::Bool, is_influence_out2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    s1 = connectivity_tracer_1_to_1(s, is_influence_out1_zero)
    s2 = connectivity_tracer_1_to_1(s, is_influence_out2_zero)
    return (s1, s2)
end

function overload_connectivity_1_to_2(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.ConnectivityTracer)
            return $SCT.connectivity_tracer_1_to_2(
                t,
                $SCT.is_influence_out1_zero_global($M.$op),
                $SCT.is_influence_out2_zero_global($M.$op),
            )
        end
    end
end

function overload_connectivity_1_to_2_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.connectivity_tracer_1_to_2(
                $SCT.tracer(d),  # TODO: add test, this was buggy
                $SCT.is_influence_out1_zero_local($M.$op, x),
                $SCT.is_influence_out2_zero_local($M.$op, x),
            )
            return ($SCT.Dual(p1_out, t1_out), $SCT.Dual(p2_out, t2_out))
        end
    end
end

## Special cases

## Exponent (requires extra types)
for S in (Integer, Rational, Irrational{:ℯ})
    Base.:^(t::ConnectivityTracer, ::S) = t
    function Base.:^(dx::D, y::S) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual(primal(dx)^y, tracer(dx))
    end

    Base.:^(::S, t::ConnectivityTracer) = t
    function Base.:^(x::S, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        return Dual(x^primal(dy), tracer(dy))
    end
end

## Rounding
Base.round(t::ConnectivityTracer, ::RoundingMode; kwargs...) = t

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = myempty(T)  # TODO: was missing Base, add tests
