# WARNING: If you are following the "Adding Overloads" guide's advice to copy an existing package extension, 
# copy another, less complicated one!
module SparseConnectivityTracerDataInterpolationsExt

if isdefined(Base, :get_extension)
    using SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using SparseConnectivityTracer: Dual, primal, tracer
    using SparseConnectivityTracer: Fill # from FillArrays.jl
    import DataInterpolations:
        LinearInterpolation,
        QuadraticInterpolation,
        LagrangeInterpolation,
        AkimaInterpolation,
        ConstantInterpolation,
        QuadraticSpline,
        CubicSpline,
        BSplineInterpolation,
        BSplineApprox
    # TODO: support when Julia 1.6 is dropped
    # CubicHermiteSpline,
    # PCHIPInterpolation,
    # QuinticHermiteSpline
else
    using ..SparseConnectivityTracer: GradientTracer, gradient_tracer_1_to_1
    using ..SparseConnectivityTracer: HessianTracer, hessian_tracer_1_to_1
    using ..SparseConnectivityTracer: Dual, primal, tracer
    using ..SparseConnectivityTracer: Fill # from FillArrays.jl
    import ..DataInterpolations:
        LinearInterpolation,
        QuadraticInterpolation,
        LagrangeInterpolation,
        AkimaInterpolation,
        ConstantInterpolation,
        QuadraticSpline,
        CubicSpline,
        BSplineInterpolation,
        BSplineApprox
    # TODO: support when Julia 1.6 is dropped
    # CubicHermiteSpline,
    # PCHIPInterpolation,
    # QuinticHermiteSpline
end

# TODO: throw errors on constructors

#========================#
# General interpolations #
#========================#

# We assume that with the exception of ConstantInterpolation and LinearInterpolation,
# all interpolations have a non-zero second derivative at some point in the input domain.

for I in (
    :QuadraticInterpolation,
    :LagrangeInterpolation,
    :AkimaInterpolation,
    :QuadraticSpline,
    :CubicSpline,
    :BSplineInterpolation,
    :BSplineApprox,
    # TODO: support when Julia 1.6 is dropped
    # :CubicHermiteSpline,
    # :QuinticHermiteSpline,
)
    # 1D Interpolations (uType<:AbstractVector)
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

    # ND Interpolations (uType<:AbstractMatrix)
    @eval function (interp::$(I){uType})(t::GradientTracer) where {uType<:AbstractMatrix}
        t = gradient_tracer_1_to_1(t, false)
        nstates = size(interp.u, 1)
        return Fill(t, nstates)
    end
    @eval function (interp::$(I){uType})(t::HessianTracer) where {uType<:AbstractMatrix}
        t = hessian_tracer_1_to_1(t, false, false)
        nstates = size(interp.u, 1)
        return Fill(t, nstates)
    end
    @eval function (interp::$(I){uType})(d::Dual) where {uType<:AbstractMatrix}
        p = interp(primal(d))
        t = interp(tracer(d))
        return Dual.(p, t)
    end
end

#=======================#
# ConstantInterpolation #
#=======================#

# 1D Interpolations (uType<:AbstractVector)
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

# ND Interpolations (uType<:AbstractMatrix)
function (interp::ConstantInterpolation{uType})(
    t::GradientTracer
) where {uType<:AbstractMatrix}
    t = gradient_tracer_1_to_1(t, true)
    nstates = size(interp.u, 1)
    return Fill(t, nstates)
end
function (interp::ConstantInterpolation{uType})(
    t::HessianTracer
) where {uType<:AbstractMatrix}
    t = hessian_tracer_1_to_1(t, true, true)
    nstates = size(interp.u, 1)
    return Fill(t, nstates)
end
function (interp::ConstantInterpolation{uType})(d::Dual) where {uType<:AbstractMatrix}
    return interp(primal(d))
end

#=====================#
# LinearInterpolation #
#=====================#

# 1D Interpolations (uType<:AbstractVector)
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
function (interp::LinearInterpolation{uType})(d::Dual) where {uType<:AbstractVector}
    p = interp(primal(d))
    t = interp(tracer(d))
    return Dual(p, t)
end

# ND Interpolations (uType<:AbstractMatrix)
function (interp::LinearInterpolation{uType})(
    t::GradientTracer
) where {uType<:AbstractMatrix}
    t = gradient_tracer_1_to_1(t, false)
    nstates = size(interp.u, 1)
    return Fill(t, nstates)
end
function (interp::LinearInterpolation{uType})(
    t::HessianTracer
) where {uType<:AbstractMatrix}
    t = hessian_tracer_1_to_1(t, false, true)
    nstates = size(interp.u, 1)
    return Fill(t, nstates)
end
function (interp::LinearInterpolation{uType})(d::Dual) where {uType<:AbstractMatrix}
    p = interp(primal(d))
    t = interp(tracer(d))
    return Dual.(p, t)
end

## Vector inputs

# TODO: 1D Interpolations (uType<:AbstractVector)

# TODO: ND Interpolations (uType<:AbstractMatrix)

end # module SparseConnectivityTracerDataInterpolationsExt
