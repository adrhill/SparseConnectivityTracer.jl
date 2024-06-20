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

# TODO: replace `conservative_or` by less conservative sparsity patterns when possible

## Determinant
LinearAlgebra.det(A::AbstractMatrix{T}) where {T<:AbstractTracer} = conservative_or(A)
LinearAlgebra.logdet(A::AbstractMatrix{T}) where {T<:AbstractTracer} = conservative_or(A)
LinearAlgebra.logabsdet(A::AbstractMatrix{T}) where {T<:AbstractTracer} = conservative_or(A)

## Norm
function LinearAlgebra.norm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return conservative_or(A)
end
function LinearAlgebra.opnorm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return conservative_or(A)
end
function LinearAlgebra.opnorm(A::AbstractMatrix{T}, p::Real=2) where {T<:AbstractTracer}
    return conservative_or(A)
end

## Eigenvalues

function LinearAlgebra.eigmax(
    A::Union{T,AbstractMatrix{T}}; permute::Bool=true, scale::Bool=true
) where {T<:AbstractTracer}
    return conservative_or(A)
end
function LinearAlgebra.eigmin(
    A::Union{T,AbstractMatrix{T}}; permute::Bool=true, scale::Bool=true
) where {T<:AbstractTracer}
    return conservative_or(A)
end
function LinearAlgebra.eigen(
    A::AbstractMatrix{T};
    permute::Bool=true,
    scale::Bool=true,
    sortby::Union{Function,Nothing}=nothing,
) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    t = conservative_or(A)
    values = Fill(t, n)
    vectors = Fill(t, n, n)
    return LinearAlgebra.Eigen(values, vectors)
end

## Inverse
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

## Division
function LinearAlgebra.:\(
    A::AbstractMatrix{T}, B::AbstractVecOrMat
) where {T<:AbstractTracer}
    Ainv = LinearAlgebra.pinv(A)
    return Ainv * B
end
