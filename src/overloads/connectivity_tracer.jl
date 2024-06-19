## 1-to-1

@noinline function connectivity_tracer_1_to_1(
    t::T, is_infl_zero::Bool
) where {T<:ConnectivityTracer}
    if is_infl_zero && !isemptytracer(t)
        return myempty(T)
    else
        return t
    end
end

function overload_connectivity_1_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::T) where {T<:$SCT.ConnectivityTracer}
            is_infl_zero = $SCT.is_infl_zero_global($M.$op)
            return $SCT.connectivity_tracer_1_to_1(t, is_infl_zero)
        end
    end
end

function overload_connectivity_1_to_1_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out = $M.$op(x)

            t = $SCT.tracer(d)
            is_infl_zero = $SCT.is_infl_zero_local($M.$op, x)
            t_out = $SCT.connectivity_tracer_1_to_1(t, is_infl_zero)
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 2-to-1

@noinline function connectivity_tracer_2_to_1(
    tx::T, ty::T, is_infl_arg1_zero::Bool, is_infl_arg2_zero::Bool
) where {T<:ConnectivityTracer}
    # TODO: add tests for isempty
    if tx.isempty && ty.isempty
        return tx # empty tracer
    elseif ty.isempty
        return connectivity_tracer_1_to_1(tx, is_infl_arg1_zero)
    elseif tx.isempty
        return connectivity_tracer_1_to_1(ty, is_infl_arg2_zero)
    else
        i_out = connectivity_tracer_2_to_1_inner(
            inputs(tx), inputs(ty), is_infl_arg1_zero, is_infl_arg2_zero
        )
        return T(i_out) # return tracer 
    end
end

function connectivity_tracer_2_to_1_inner(
    sx::S, sy::S, is_infl_arg1_zero::Bool, is_infl_arg2_zero::Bool
) where {S<:AbstractSet{<:Integer}}
    if is_infl_arg1_zero && is_infl_arg2_zero
        return myempty(S)
    elseif !is_infl_arg1_zero && is_infl_arg2_zero
        return sx
    elseif is_infl_arg1_zero && !is_infl_arg2_zero
        return sy
    else
        return union(sx, sy) # return set
    end
end

function overload_connectivity_2_to_1(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(tx::T, ty::T) where {T<:$SCT.ConnectivityTracer}
            is_infl_arg1_zero = $SCT.is_infl_arg1_zero_global($M.$op)
            is_infl_arg2_zero = $SCT.is_infl_arg2_zero_global($M.$op)
            return $SCT.connectivity_tracer_2_to_1(
                tx, ty, is_infl_arg1_zero, is_infl_arg2_zero
            )
        end

        function $M.$op(tx::$SCT.ConnectivityTracer, ::Real)
            is_infl_arg1_zero = $SCT.is_infl_arg1_zero_global($M.$op)
            return $SCT.connectivity_tracer_1_to_1(tx, is_infl_arg1_zero)
        end

        function $M.$op(::Real, ty::$SCT.ConnectivityTracer)
            is_infl_arg2_zero = $SCT.is_infl_arg2_zero_global($M.$op)
            return $SCT.connectivity_tracer_1_to_1(ty, is_infl_arg2_zero)
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

            tx = $SCT.tracer(dx)
            ty = $SCT.tracer(dy)
            is_infl_arg1_zero = $SCT.is_infl_arg1_zero_local($M.$op, x, y)
            is_infl_arg2_zero = $SCT.is_infl_arg2_zero_local($M.$op, x, y)
            t_out = $SCT.connectivity_tracer_2_to_1(
                tx, ty, is_infl_arg1_zero, is_infl_arg2_zero
            )
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            dx::D, y::Real
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(dx)
            p_out = $M.$op(x, y)

            tx = $SCT.tracer(dx)
            is_infl_arg1_zero = $SCT.is_infl_arg1_zero_local($M.$op, x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(tx, is_infl_arg1_zero)
            return $SCT.Dual(p_out, t_out)
        end

        function $M.$op(
            x::Real, dy::D
        ) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            y = $SCT.primal(dy)
            p_out = $M.$op(x, y)

            ty = $SCT.tracer(dy)
            is_infl_arg2_zero = $SCT.is_infl_arg2_zero_local($M.$op, x, y)
            t_out = $SCT.connectivity_tracer_1_to_1(ty, is_infl_arg2_zero)
            return $SCT.Dual(p_out, t_out)
        end
    end
end

## 1-to-2

@noinline function connectivity_tracer_1_to_2(
    t::T, is_infl_out1_zero::Bool, is_infl_out2_zero::Bool
) where {T<:ConnectivityTracer}
    if isemptytracer(t) # TODO: add test
        return (t, t)
    else
        t_out1 = connectivity_tracer_1_to_1(t, is_infl_out1_zero)
        t_out2 = connectivity_tracer_1_to_1(t, is_infl_out2_zero)
        return (t_out1, t_out2) # return tracers 
    end
end

function overload_connectivity_1_to_2(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(t::$SCT.ConnectivityTracer)
            is_infl_out1_zero = $SCT.is_infl_out1_zero_global($M.$op)
            is_infl_out2_zero = $SCT.is_infl_out2_zero_global($M.$op)
            return $SCT.connectivity_tracer_1_to_2(t, is_infl_out1_zero, is_infl_out2_zero)
        end
    end
end

function overload_connectivity_1_to_2_dual(M, op)
    SCT = SparseConnectivityTracer
    return quote
        function $M.$op(d::D) where {P,T<:$SCT.ConnectivityTracer,D<:$SCT.Dual{P,T}}
            x = $SCT.primal(d)
            p_out1, p_out2 = $M.$op(x)

            t = tracer(d)
            is_infl_out1_zero = $SCT.is_infl_out1_zero_local($M.$op, x)
            is_infl_out2_zero = $SCT.is_infl_out2_zero_local($M.$op, x)
            t_out1, t_out2 = $SCT.connectivity_tracer_1_to_2(
                t, is_infl_out1_zero, is_infl_out2_zero
            )# TODO: add test, this was buggy
            return ($SCT.Dual(p_out1, t_out1), $SCT.Dual(p_out2, t_out2))
        end
    end
end

## Special cases

## Exponent (requires extra types)
for S in (Integer, Rational, Irrational{:ℯ})
    Base.:^(t::ConnectivityTracer, ::S) = t
    Base.:^(::S, t::ConnectivityTracer) = t
    function Base.:^(dx::D, y::S) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(dx)
        return Dual(x^y, tracer(dx))
    end
    function Base.:^(x::S, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        y = primal(dy)
        return Dual(x^y, tracer(dy))
    end
end

## Rounding
Base.round(t::ConnectivityTracer, ::RoundingMode; kwargs...) = t

## Random numbers
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = myempty(T)  # TODO: was missing Base, add tests
