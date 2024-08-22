module SparseConnectivityTracerDataInterpolationsExt

if isdefined(Base, :get_extension)
    using SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using SparseConnectivityTracer: Dual, primal, tracer
    import DataInterpolations:
        ConstantInterpolation, LinearInterpolation, AbstractInterpolation
else
    using ..SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using ..SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using ..SparseConnectivityTracer: Dual, primal, tracer
    import ..DataInterpolations:
        ConstantInterpolation, LinearInterpolation, AbstractInterpolation
end

# We assume with the exception of ConstantInterpolation and LinearInterpolation,
# all interpolations have a non-zero second derivative at some point in the input domain.

(interp::AbstractInterpolation)(t::GradientTracer) = gradient_tracer_1_to_1(t, false)
(interp::ConstantInterpolation)(t::GradientTracer) = gradient_tracer_1_to_1(t, true)

(interp::AbstractInterpolation)(t::HessianTracer) = hessian_tracer_1_to_1(t, false, false)
(interp::LinearInterpolation)(t::HessianTracer) = hessian_tracer_1_to_1(t, false, true)
(interp::ConstantInterpolation)(t::HessianTracer) = hessian_tracer_1_to_1(t, true, true)

function (interp::AbstractInterpolation)(d::Dual)
    p = interp(primal(d))
    t = interp(tracer(d))
    return Dual(p, t)
end
(interp::ConstantInterpolation)(d::Dual) = interp(primal(d))

end # module SparseConnectivityTracerDataInterpolationsExt
