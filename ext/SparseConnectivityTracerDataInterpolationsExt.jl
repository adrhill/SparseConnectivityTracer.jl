module SparseConnectivityTracerDataInterpolationsExt
import SparseConnectivityTracer as SCT

if isdefined(Base, :get_extension)
    using DataInterpolations
else
    using ..DataInterpolations
end

interpolation_types = []
for name in names(DataInterpolations)
    if isdefined(DataInterpolations, name)
        val = getfield(DataInterpolations, name)
        if val isa Type && val <: DataInterpolations.AbstractInterpolation
            push!(interpolation_types, val)
        end
    end
end

for interpolation_type in interpolation_types
    if interpolation_type == ConstantInterpolation
        @eval SCT.is_der1_zero_global(::Type{$interpolation_type}) = true
        @eval SCT.is_der2_zero_global(::Type{$interpolation_type}) = true
    elseif interpolation_type == LinearInterpolation
        @eval SCT.is_der1_zero_global(::Type{$interpolation_type}) = false
        @eval SCT.is_der2_zero_global(::Type{$interpolation_type}) = true
    else
        @eval SCT.is_der1_zero_global(::Type{$interpolation_type}) = false
        @eval SCT.is_der2_zero_global(::Type{$interpolation_type}) = false
    end
end

SCT.overload_gradient_1_to_1(:DataInterpolations, interpolation_types)

end # module SparseConnectivityTracerDataInterpolationsExt
