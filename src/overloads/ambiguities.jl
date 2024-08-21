## Special overloads to avoid ambiguity errors
for S in (Integer, Rational, Irrational{:ℯ})
    Base.:^(t::T, ::S) where {T<:GradientTracer} = t
    Base.:^(::S, t::T) where {T<:GradientTracer} = t
    Base.:^(t::T, ::S) where {T<:HessianTracer} = hessian_tracer_1_to_1(t, false, false)
    Base.:^(::S, t::T) where {T<:HessianTracer} = hessian_tracer_1_to_1(t, false, false)

    function Base.:^(d::D, y::S) where {P,T<:GradientTracer,D<:Dual{P,T}}
        x = primal(d)
        t = gradient_tracer_1_to_1(tracer(d), false)
        return Dual(x^y, t)
    end
    function Base.:^(x::S, d::D) where {P,T<:GradientTracer,D<:Dual{P,T}}
        y = primal(d)
        t = gradient_tracer_1_to_1(tracer(d), false)
        return Dual(x^y, t)
    end

    function Base.:^(d::D, y::S) where {P,T<:HessianTracer,D<:Dual{P,T}}
        x = primal(d)
        t = hessian_tracer_1_to_1(tracer(d), false, false)
        return Dual(x^y, t)
    end
    function Base.:^(x::S, d::D) where {P,T<:HessianTracer,D<:Dual{P,T}}
        y = primal(d)
        t = hessian_tracer_1_to_1(tracer(d), false, false)
        return Dual(x^y, t)
    end
end

for TT in (GradientTracer, HessianTracer)
    function Base.isless(dx::D, y::AbstractFloat) where {P<:Real,T<:TT,D<:Dual{P,T}}
        return isless(primal(dx), y)
    end
end
