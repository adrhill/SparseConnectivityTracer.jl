"""
    conservative_or(tracers...)

Compute the most conservative elementwise OR of sparsity patterns. 
"""
function conservative_or(ts::AbstractArray{T}) where {T<:AbstractTracer}
    # TODO: improve performance
    return reduce(conservative_or, ts; init=myempty(T))
end

function conservative_or(a::T, b::T) where {T<:ConnectivityTracer}
    return connectivity_tracer_2_to_1(a, b, false, false)
end
function conservative_or(a::T, b::T) where {T<:GradientTracer}
    return gradient_tracer_2_to_1(a, b, false, false)
end
function conservative_or(a::T, b::T) where {T<:HessianTracer}
    return hessian_tracer_2_to_1(a, b, false, false, false, false, false)
end
