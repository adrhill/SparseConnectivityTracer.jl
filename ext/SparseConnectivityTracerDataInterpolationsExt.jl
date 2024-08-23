module SparseConnectivityTracerDataInterpolationsExt

if isdefined(Base, :get_extension)
    using SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using SparseConnectivityTracer: Dual, primal, tracer
    import DataInterpolations:
        LinearInterpolation,
        QuadraticInterpolation,
        LagrangeInterpolation,
        AkimaInterpolation,
        ConstantInterpolation,
        QuadraticSpline,
        CubicSpline,
        BSplineInterpolation,
        BSplineApprox,
        CubicHermiteSpline,
        PCHIPInterpolation,
        QuinticHermiteSpline
else
    using ..SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using ..SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using ..SparseConnectivityTracer: Dual, primal, tracer
    import ..DataInterpolations:
        LinearInterpolation,
        QuadraticInterpolation,
        LagrangeInterpolation,
        AkimaInterpolation,
        ConstantInterpolation,
        QuadraticSpline,
        CubicSpline,
        BSplineInterpolation,
        BSplineApprox,
        CubicHermiteSpline,
        PCHIPInterpolation,
        QuinticHermiteSpline
end

## ConstantInterpolation
function (interp::ConstantInterpolation{uType})(
    t::GradientTracer
) where {uType<:AbstractVector}
    return gradient_tracer_1_to_1(t, true)
end
function (interp::ConstantInterpolation{uType})(
    t::HessianTracer
) where {uType<:AbstractVector}
    return hessian_tracer_1_to_1(t, true, true)
end
function (interp::ConstantInterpolation{uType})(d::Dual) where {uType<:AbstractVector}
    return interp(primal(d))
end

## LinearInterpolation
function (interp::LinearInterpolation{uType})(
    t::GradientTracer
) where {uType<:AbstractVector}
    return gradient_tracer_1_to_1(t, false)
end
function (interp::LinearInterpolation{uType})(
    t::HessianTracer
) where {uType<:AbstractVector}
    return hessian_tracer_1_to_1(t, false, true)
end

## We assume that all other interpolations have a non-zero second derivative at some point in the input domain.
for I in (
    :QuadraticInterpolation,
    :LagrangeInterpolation,
    :AkimaInterpolation,
    :QuadraticSpline,
    :CubicSpline,
    :BSplineInterpolation,
    :BSplineApprox,
    :CubicHermiteSpline,
    :QuinticHermiteSpline,
)
    @eval function (interp::$(I){uType})(t::GradientTracer) where {uType<:AbstractVector}
        return gradient_tracer_1_to_1(t, false)
    end
    @eval function (interp::$(I){uType})(t::HessianTracer) where {uType<:AbstractVector}
        return hessian_tracer_1_to_1(t, false, false)
    end
    @eval function (interp::$(I){uType})(d::Dual) where {uType<:AbstractVector}
        p = interp(primal(d))
        t = interp(tracer(d))
        return Dual(p, t)
    end
end

end # module SparseConnectivityTracerDataInterpolationsExt
