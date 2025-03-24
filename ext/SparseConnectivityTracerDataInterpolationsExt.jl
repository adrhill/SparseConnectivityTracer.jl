# WARNING: If you are following the "Adding Overloads" guide's advice to copy an existing package extension,
# copy another, less complicated one!
module SparseConnectivityTracerDataInterpolationsExt

import SparseConnectivityTracer as SCT
using SparseConnectivityTracer: AbstractTracer, Dual, primal, tracer, get_output_dim
using SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
using SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
using FillArrays: Fill # from FillArrays.jl
import DataInterpolations: AbstractInterpolation
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
    # PCHIPInterpolation,
    QuinticHermiteSpline

#===========#
# Utilities #
#===========#

# Get output dimension for parameterizing AbstractInterpolations
# This function was removed together with the output type parameter
# in DataInterpolations v8: https://github.com/SciML/DataInterpolations.jl/pull/396
# TODO use DataInterpolations.output_dim instead, which always returns an integer
# which is 0 for scalar output.
function SCT.get_output_dim(u::AbstractVector{<:Number})
    return (1,)
end

function SCT.get_output_dim(u::AbstractVector)
    return (length(first(u)),)
end

function SCT.get_output_dim(u::AbstractArray)
    return size(u)[1:(end - 1)]
end

function _sct_interpolate(
    ::AbstractInterpolation{T},
    uType::Type{V},
    t::GradientTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,V<:AbstractVector}
    return gradient_tracer_1_to_1(t, is_der_1_zero)
end
function _sct_interpolate(
    ::AbstractInterpolation{T},
    uType::Type{V},
    t::HessianTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,V<:AbstractVector}
    return hessian_tracer_1_to_1(t, is_der_1_zero, is_der_2_zero)
end
function _sct_interpolate(
    interp::AbstractInterpolation{T},
    uType::Type{M},
    t::GradientTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,M<:AbstractMatrix}
    t = gradient_tracer_1_to_1(t, is_der_1_zero)
    N = get_output_dim(interp.u)
    return Fill(t, N)
end
function _sct_interpolate(
    interp::AbstractInterpolation{T},
    uType::Type{M},
    t::HessianTracer,
    is_der_1_zero,
    is_der_2_zero,
) where {T,M<:AbstractMatrix}
    t = hessian_tracer_1_to_1(t, is_der_1_zero, is_der_2_zero)
    N = get_output_dim(interp.u)
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
