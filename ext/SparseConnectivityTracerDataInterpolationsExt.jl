# WARNING: If you are following the "Adding Overloads" guide's advice to copy an existing package extension,
# copy another, less complicated one!
module SparseConnectivityTracerDataInterpolationsExt

using SparseConnectivityTracer: AbstractTracer, Dual, primal, tracer
using SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
using SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
using FillArrays: Fill # from FillArrays.jl
import DataInterpolations: AbstractInterpolation
import DataInterpolations:
    LinearInterpolation,
    SmoothedLinearInterpolation,
    QuadraticInterpolation,
    LagrangeInterpolation,
    AkimaInterpolation,
    ConstantInterpolation,
    QuadraticSpline,
    CubicSpline,
    BSplineInterpolation,
    BSplineApprox,
    CubicHermiteSpline,
    # PCHIPInterpolation,
    QuinticHermiteSpline

#===========#
# Utilities #
#===========#

function _sct_interpolate(
    ::AbstractInterpolation{T,N},
    uType::Type{V},
    t::GradientTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,N,V<:AbstractVector}
    return gradient_tracer_1_to_1(t, is_der_1_zero)
end
function _sct_interpolate(
    ::AbstractInterpolation{T,N},
    uType::Type{V},
    t::HessianTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,N,V<:AbstractVector}
    return hessian_tracer_1_to_1(t, is_der_1_zero, is_der_2_zero)
end
function _sct_interpolate(
    ::AbstractInterpolation{T,N},
    uType::Type{M},
    t::GradientTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,N,M<:AbstractMatrix}
    t = gradient_tracer_1_to_1(t, is_der_1_zero)
    return Fill(t, N)
end
function _sct_interpolate(
    ::AbstractInterpolation{T,N},
    uType::Type{M},
    t::HessianTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,N,M<:AbstractMatrix}
    t = hessian_tracer_1_to_1(t, is_der_1_zero, is_der_2_zero)
    return Fill(t, N)
end

#===========#
# Overloads #
#===========#

# We assume that with the exception of ConstantInterpolation and LinearInterpolation,
# all interpolations have a non-zero second derivative at some point in the input domain.

for (I, is_der1_zero, is_der2_zero) in (
    (:ConstantInterpolation, true, true),
    (:LinearInterpolation, false, true),
    (:SmoothedLinearInterpolation, false, false),
    (:QuadraticInterpolation, false, false),
    (:LagrangeInterpolation, false, false),
    (:AkimaInterpolation, false, false),
    (:QuadraticSpline, false, false),
    (:CubicSpline, false, false),
    (:BSplineInterpolation, false, false),
    (:BSplineApprox, false, false),
    (:CubicHermiteSpline, false, false),
    (:QuinticHermiteSpline, false, false),
)
    @eval function (interp::$(I){uType})(t::AbstractTracer) where {uType}
        return _sct_interpolate(interp, uType, t, $is_der1_zero, $is_der2_zero)
    end
end

# Some Interpolations require custom overloads on `Dual` due to mutation of caches.
for I in (
    :LagrangeInterpolation,
    :BSplineInterpolation,
    :BSplineApprox,
    :CubicHermiteSpline,
    :QuinticHermiteSpline,
)
    @eval function (interp::$(I){uType})(d::Dual) where {uType<:AbstractVector}
        p = interp(primal(d))
        t = interp(tracer(d))
        return Dual(p, t)
    end

    @eval function (interp::$(I){uType})(d::Dual) where {uType<:AbstractMatrix}
        p = interp(primal(d))
        t = interp(tracer(d))
        return Dual.(p, t)
    end
end

end # module SparseConnectivityTracerDataInterpolationsExt
