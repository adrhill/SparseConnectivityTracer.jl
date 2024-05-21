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

for fn in nameof.(ops_1_to_1)
    @eval function Base.$fn(t::T) where {T<:ConnectivityTracer}
        return connectivity_tracer_1_to_1(t, is_influence_zero_global($fn))
    end
    @eval function Base.$fn(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(d)
        p_out = Base.$fn(x)
        t_out = connectivity_tracer_1_to_1(tracer(d), is_influence_zero_local($fn, x))
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
    else # ∂f∂x ≠ 0 
        if is_influence_arg2_zero
            return tx
        else
            return T(inputs(tx) ∪ inputs(ty))
        end
    end
end

for fn in nameof.(ops_2_to_1)
    @eval function Base.$fn(tx::T, ty::T) where {T<:ConnectivityTracer}
        return connectivity_tracer_2_to_1(
            tx, ty, is_influence_arg1_zero_global($fn), is_influence_arg2_zero_global($fn)
        )
    end
    @eval function Base.$fn(dx::D, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(dx)
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = connectivity_tracer_2_to_1(
            tracer(dx),
            tracer(dy),
            is_influence_arg1_zero_local($fn, x, y),
            is_influence_arg2_zero_local($fn, x, y),
        )
        return Dual(p_out, t_out)
    end

    @eval function Base.$fn(tx::ConnectivityTracer, ::Number)
        return connectivity_tracer_1_to_1(tx, is_influence_arg1_zero_global($fn))
    end
    @eval function Base.$fn(dx::D, y::Number) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(dx)
        p_out = Base.$fn(x, y)
        t_out = connectivity_tracer_1_to_1(
            tracer(dx), is_influence_arg1_zero_local($fn, x, y)
        )
        return Dual(p_out, t_out)
    end

    @eval function Base.$fn(::Number, ty::ConnectivityTracer)
        return connectivity_tracer_1_to_1(ty, is_influence_arg2_zero_global($fn))
    end
    @eval function Base.$fn(x::Number, dy::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        y = primal(dy)
        p_out = Base.$fn(x, y)
        t_out = connectivity_tracer_1_to_1(
            tracer(dy), is_influence_arg2_zero_local($fn, x, y)
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

for fn in nameof.(ops_1_to_2)
    @eval function Base.$fn(t::ConnectivityTracer)
        return connectivity_tracer_1_to_2(
            t, is_influence_out1_zero_global($fn), is_influence_out2_zero_global($fn)
        )
    end

    @eval function Base.$fn(d::D) where {P,T<:ConnectivityTracer,D<:Dual{P,T}}
        x = primal(d)
        p1_out, p2_out = Base.$fn(x)
        t1_out, t2_out = connectivity_tracer_1_to_2(
            t, is_influence_out1_zero_local($fn, x), is_influence_out2_zero_local($fn, x)
        )
        return (Dual(p1_out, t1_out), Dual(p1_out, t1_out))
    end
end

# Extra types required for exponent
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
rand(::AbstractRNG, ::SamplerType{T}) where {T<:ConnectivityTracer} = empty(T)
