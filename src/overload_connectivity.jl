## 1-to-1

@noinline function connectivity_tracer_1_to_1(
    t::T, is_influence_zero::Bool
) where {T<:ConnectivityTracer}
    if t.isempty # TODO: add test
        return t
    else
        inputs = connectivity_tracer_1_to_1_inner(t.inputs, is_influence_zero)
        return T(inputs) # return tracer
    end
end

function connectivity_tracer_1_to_1_inner(
    s::S, is_influence_zero::Bool
) where {S<:AbstractSet}
    if is_influence_zero
        return myempty(S)
    else
        return s # return set
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
            x = d.primal
            p_out = $M.$op(x)
            t_out = $SCT.connectivity_tracer_1_to_1(
                d.tracer, $SCT.is_influence_zero_local($M.$op, x)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

@noinline function connectivity_tracer_2_to_1(
    tx::T, ty::T, is_influence_arg1_zero::Bool, is_influence_arg2_zero::Bool
) where {T<:ConnectivityTracer}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return connectivity_tracer_1_to_1(tx, is_influence_arg1_zero)
    elseif tx.isempty
        return connectivity_tracer_1_to_1(ty, is_influence_arg2_zero)
    else
        inputs = connectivity_tracer_2_to_1_inner(
            tx.inputs, ty.inputs, is_influence_arg1_zero, is_influence_arg2_zero
        )
        return T(inputs) # return tracer 
    end
end

function connectivity_tracer_2_to_1_inner(
    sx::S, sy::S, is_influence_arg1_zero::Bool, is_influence_arg2_zero::Bool
) where {S<:AbstractSet}
    if is_influence_arg1_zero && is_influence_arg2_zero
        return myempty(S)
    elseif !is_influence_arg1_zero && is_influence_arg2_zero
        return sx
    elseif is_influence_arg1_zero && !is_influence_arg2_zero
        return sy
    else
        return union(sx, sy) # return set
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
            x = dx.primal
            y = dy.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_2_to_1(
                dx.tracer,
                dy.tracer,
                $SCT.is_influence_arg1_zero_local($M.$op, x, y),
                $SCT.is_influence_arg2_zero_local($M.$op, x, y),
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            dx::D, y::Real
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = dx.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(
                dx.tracer, $SCT.is_influence_arg1_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            x::Real, dy::D
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            y = dy.primal
            p_out = $M.$op(x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(
                dy.tracer, $SCT.is_influence_arg2_zero_local($M.$op, x, y)
            )
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

@noinline function connectivity_tracer_1_to_2(
    t::T, is_influence_out1_zero::Bool, is_influence_out2_zero::Bool
) where {T<:ConnectivityTracer}
    if t.isempty # TODO: add test
        return (t, t)
    else
        inputs1, inputs2 = connectivity_tracer_1_to_2_inner(
            t.inputs, is_influence_out1_zero, is_influence_out2_zero
        )
        return (T(inputs1), T(inputs2)) # return tracers 
    end
end

function connectivity_tracer_1_to_2_inner(
    s::S, is_influence_out1_zero::Bool, is_influence_out2_zero::Bool
) where {S<:AbstractSet}
    s1 = connectivity_tracer_1_to_1_inner(s, is_influence_out1_zero)
    s2 = connectivity_tracer_1_to_1_inner(s, is_influence_out2_zero)
    return (s1, s2) # return sets
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
            x = d.primal
            p1_out, p2_out = $M.$op(x)
            t1_out, t2_out = $SCT.connectivity_tracer_1_to_2(
                d.tracer,  # TODO: add test, this was buggy
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
    Base.:^(::S, t::ConnectivityTracer) = t
    function Base.:^(dx::D, y::S) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = dx.primal
        return Dual(x^y, dx.tracer)
    end
    function Base.:^(x::S, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        y = dy.primal
        return Dual(x^y, dy.tracer)
    end
end

## Rounding
Base.round(t::ConnectivityTracer, ::RoundingMode; kwargs...) = t

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = myempty(T)  # TODO: was missing Base, add tests
