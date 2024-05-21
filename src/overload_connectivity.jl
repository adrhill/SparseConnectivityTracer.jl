## 1-to-1

function connectivity_tracer_1_to_1(
    t::T, is_influence_zero::Bool
) where {T<:ConnectivityTracer}
    if is_influence_zero
        return empty(T)
    else
        return t
    end
end

function overload_connectivity_1_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(t::T) where {T<:ConnectivityTracer}
        return connectivity_tracer_1_to_1(t, is_influence_zero_global($ms.$fns))
    end
    @eval function $ms.$fns(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(d)
        p_out = $ms.$fns(x)
        t_out = connectivity_tracer_1_to_1(tracer(d), is_influence_zero_local($ms.$fns, x))
        return Dual(p_out, t_out)
    end
end

## 2-to-1

function connectivity_tracer_2_to_1(
    tx::T, ty::T, is_influence_arg1_zero::Bool, is_influence_arg2_zero::Bool
) where {T<:ConnectivityTracer}
    if is_influence_arg1_zero
        if is_influence_arg2_zero
            return empty(T)
        else
            return ty
        end
    else # x -> f ≠ 0 
        if is_influence_arg2_zero
            return tx
        else
            return T(inputs(tx) ∪ inputs(ty))
        end
    end
end

function overload_connectivity_2_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(tx::T, ty::T) where {T<:ConnectivityTracer}
        return connectivity_tracer_2_to_1(
            tx,
            ty,
            is_influence_arg1_zero_global($ms.$fns),
            is_influence_arg2_zero_global($ms.$fns),
        )
    end
    @eval function $ms.$fns(dx::D, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(dx)
        y = primal(dy)
        p_out = $ms.$fns(x, y)
        t_out = connectivity_tracer_2_to_1(
            tracer(dx),
            tracer(dy),
            is_influence_arg1_zero_local($ms.$fns, x, y),
            is_influence_arg2_zero_local($ms.$fns, x, y),
        )
        return Dual(p_out, t_out)
    end

    @eval function $ms.$fns(tx::ConnectivityTracer, ::Number)
        return connectivity_tracer_1_to_1(tx, is_influence_arg1_zero_global($fns))
    end
    @eval function $ms.$fns(dx::D, y::Number) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(dx)
        p_out = $ms.$fns(x, y)
        t_out = connectivity_tracer_1_to_1(
            tracer(dx), is_influence_arg1_zero_local($ms.$fns, x, y)
        )
        return Dual(p_out, t_out)
    end

    @eval function $ms.$fns(::Number, ty::ConnectivityTracer)
        return connectivity_tracer_1_to_1(ty, is_influence_arg2_zero_global($fns))
    end
    @eval function $ms.$fns(x::Number, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        y = primal(dy)
        p_out = $ms.$fns(x, y)
        t_out = connectivity_tracer_1_to_1(
            tracer(dy), is_influence_arg2_zero_local($ms.$fns, x, y)
        )
        return Dual(p_out, t_out)
    end
end

## 1-to-2

function connectivity_tracer_1_to_2(
    t::T, is_influence_out1_zero::Bool, is_influence_out2_zero::Bool
) where {T<:ConnectivityTracer}
    t1 = connectivity_tracer_1_to_1(t, is_influence_out1_zero)
    t2 = connectivity_tracer_1_to_1(t, is_influence_out2_zero)
    return (t1, t2)
end

function overload_connectivity_1_to_2(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval function $ms.$fns(t::ConnectivityTracer)
        return connectivity_tracer_1_to_2(
            t,
            is_influence_out1_zero_global($ms.$fns),
            is_influence_out2_zero_global($ms.$fns),
        )
    end

    @eval function $ms.$fns(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(d)
        p1_out, p2_out = $ms.$fns(x)
        t1_out, t2_out = connectivity_tracer_1_to_2(
            t,
            is_influence_out1_zero_local($ms.$fns, x),
            is_influence_out2_zero_local($ms.$fns, x),
        )
        return (Dual(p1_out, t1_out), Dual(p2_out, t2_out))
    end
end

## Actual overloads

for op in ops_1_to_1
    overload_connectivity_1_to_1(Base, op)
end

for op in ops_2_to_1
    overload_connectivity_2_to_1(Base, op)
end

for op in ops_1_to_2
    overload_connectivity_1_to_2(Base, op)
end

## Special cases

## Exponent (requires extra types)

for S in (Real, Integer, Rational, Irrational{:ℯ})
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
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = empty(T)  # TODO: was missing Base, add tests
