"""
    conservative_or(tracers...)

Compute the most conservative elementwise OR of tracer sparsity patterns. 
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

#==================#
# LinearAlgebra.jl #
#==================#

function LinearAlgebra.det(A::AbstractMatrix{T}) where {T<:AbstractTracer}
    return conservative_or(A)
end

function LinearAlgebra.logdet(A::AbstractMatrix{T}) where {T<:AbstractTracer}
    return conservative_or(A)
end

function LinearAlgebra.inv(A::StridedMatrix{T}) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    t = conservative_or(A)
    return Fill(t, size(A)...)
end

function LinearAlgebra.pinv(
    A::AbstractMatrix{T}; atol::Real=0.0, rtol::Real=0.0
) where {T<:AbstractTracer}
    n, m = size(A)
    t = conservative_or(A)
    return Fill(t, m, n)
end

function LinearAlgebra.:\(
    A::AbstractMatrix{T}, B::AbstractVecOrMat
) where {T<:AbstractTracer}
    Ainv = LinearAlgebra.pinv(A)
    return Ainv * B
end
