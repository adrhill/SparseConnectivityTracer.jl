module SparseConnectivityTracerDataInterpolationsExt
import SparseConnectivityTracer as SCT

if isdefined(Base, :get_extension)
    using DataInterpolations
    using DataInterpolations: _interpolate, _derivative
else
    using ..DataInterpolations
    using ..DataInterpolations: _interpolate, _derivative
end

operations = [_interpolate, _derivative]

for operation in operations
    @eval SCT.is_der1_arg1_zero_global(::typeof($operation)) = true
    @eval SCT.is_der2_arg1_zero_global(::typeof($operation)) = true
    @eval SCT.is_der1_arg2_zero_global(::typeof($operation)) = false
    @eval SCT.is_der2_arg2_zero_global(::typeof($operation)) = false
    @eval SCT.is_der_cross_zero_global(::typeof($operation)) = true
end

eval(SCT.overload_gradient_2_to_1(:DataInterpolations, operations))
eval(SCT.overload_hessian_2_to_1(:DataInterpolations, operations))

end # module SparseConnectivityTracerDataInterpolationsExt
