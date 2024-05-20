## 1-to-1

function gradient_tracer_1_to_1(t::T, is_firstder_zero::Bool) where {T<:GradientTracer}
    if is_firstder_zero
        return empty(T)
    else
        return t
    end
end

function overload_gradient_1_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval begin
        function $ms.$fns(t::GradientTracer)
            return gradient_tracer_1_to_1(t, is_firstder_zero_global($ms.$fns))
        end
        function $ms.$fns(d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
            x = primal(d)
            p_out = $ms.$fns(x)
            t_out = gradient_tracer_1_to_1(tracer(d), is_firstder_zero_local($fns, x))
            return Dual(p_out, t_out)
        end
    end
end

## 2-to-1

function gradient_tracer_2_to_1(
    tx::T, ty::T, is_firstder_arg1_zero::Bool, is_firstder_arg2_zero::Bool
) where {T<:GradientTracer}
    if is_firstder_arg1_zero
        if is_firstder_arg2_zero
            return empty(T)
        else
            return ty
        end
    else # ∂f∂x ≠ 0 
        if is_firstder_arg2_zero
            return tx
        else
            return T(gradient(tx) ∪ gradient(ty))
        end
    end
end

function overload_gradient_2_to_1(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval begin
        function $ms.$fns(tx::T, ty::T) where {T<:GradientTracer}
            return gradient_tracer_2_to_1(
                tx,
                ty,
                is_firstder_arg1_zero_global($ms.$fns),
                is_firstder_arg2_zero_global($ms.$fns),
            )
        end
        function $ms.$fns(dx::D, dy::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
            x = primal(dx)
            y = primal(dy)
            p_out = $ms.$fns(x, y)
            t_out = gradient_tracer_2_to_1(
                tracer(dx),
                tracer(dy),
                is_firstder_arg1_zero_local($ms.$fns, x, y),
                is_firstder_arg2_zero_local($ms.$fns, x, y),
            )
            return Dual(p_out, t_out)
        end

        function $ms.$fns(tx::GradientTracer, ::Number)
            return gradient_tracer_1_to_1(tx, is_firstder_arg1_zero_global($ms.$fns))
        end
        function $ms.$fns(dx::D, y::Number) where {P,T<:GradientTracer,D<:Dual{P,T}}
            x = primal(dx)
            p_out = $ms.$fns(x, y)
            t_out = gradient_tracer_1_to_1(
                tracer(dx), is_firstder_arg1_zero_local($ms.$fns, x, y)
            )
            return Dual(p_out, t_out)
        end

        function $ms.$fns(::Number, ty::GradientTracer)
            return gradient_tracer_1_to_1(ty, is_firstder_arg2_zero_global($ms.$fns))
        end
        function $ms.$fns(x::Number, dy::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
            y = primal(dy)
            p_out = $ms.$fns(x, y)
            t_out = gradient_tracer_1_to_1(
                tracer(dy), is_firstder_arg2_zero_local($ms.$fns, x, y)
            )
            return Dual(p_out, t_out)
        end
    end
end

## 1-to-2

function gradient_tracer_1_to_2(
    t::T, is_firstder_out1_zero::Bool, is_firstder_out2_zero::Bool
) where {T<:GradientTracer}
    t1 = gradient_tracer_1_to_1(t, is_firstder_out1_zero)
    t2 = gradient_tracer_1_to_1(t, is_firstder_out2_zero)
    return (t1, t2)
end

function overload_gradient_1_to_2(m::Module, fn::Function)
    ms, fns = nameof(m), nameof(fn)
    @eval begin
        function $ms.$fns(t::GradientTracer)
            return gradient_tracer_1_to_2(
                t,
                is_firstder_out1_zero_global($ms.$fns),
                is_firstder_out2_zero_global($ms.$fns),
            )
        end

        function $ms.$fns(d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
            x = primal(d)
            p1_out, p2_out = $ms.$fns(x)
            t1_out, t2_out = gradient_tracer_1_to_2(
                t,
                is_firstder_out1_zero_local($ms.$fns, x),
                is_firstder_out2_zero_local($ms.$fns, x),
            )
            return (Dual(p1_out, t1_out), Dual(p2_out, t2_out))  # TODO: this was wrong, add test
        end
    end
end

## Actual overloads

for op in ops_1_to_1
    overload_gradient_1_to_1(Base, op)
end

for op in ops_2_to_1
    overload_gradient_2_to_1(Base, op)
end

for op in ops_1_to_2
    overload_gradient_1_to_2(Base, op)
end

## Special cases

## Exponent (requires extra types)

for S in (Real, Integer, Rational, Irrational{:ℯ})
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
Base.round(t::T, ::RoundingMode; kwargs...) where {T<:GradientTracer} = empty(T)
function Base.round(
    d::D, mode::RoundingMode; kwargs...
) where {P,T<:GradientTracer,D<:Dual{P,T}}
    return Dual(round(primal(d), mode; kwargs...), empty(T))
end

## Random numbers 
Base.rand(::AbstractRNG, ::SamplerType{T}) where {T<:GradientTracer} = empty(T)  # TODO: was missing Base, add tests
