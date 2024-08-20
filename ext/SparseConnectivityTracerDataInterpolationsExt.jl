module SparseConnectivityTracerDataInterpolationsExt

if isdefined(Base, :get_extension)
    import SparseConnectivityTracer as SCT
    using DataInterpolations
else
    import ..SparseConnectivitytracer as SCT
    using ..DataInterpolations
end

# In general the first and second derivatives are non-zero
SCT.is_der1_zero_global(::DataInterpolations.AbstractInterpolation) = false
SCT.is_der2_zero_global(::DataInterpolations.AbstractInterpolation) = false

# Special cases
SCT.is_der1_zero_global(::ConstantInterpolation) = true
SCT.is_der2_zero_global(::ConstantInterpolation) = true
SCT.is_der2_zero_global(::LinearInterpolation) = true

# To do: derivative, integral

eval(SCT.overload_gradient_1_to_1(:DataInterpolations, AbstractInterpolation))

end # module SparseConnectivityTracerDataInterpolationsExt