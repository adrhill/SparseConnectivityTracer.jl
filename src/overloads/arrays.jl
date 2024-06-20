"""
    second_order_or(tracers)

Compute the most conservative elementwise OR of tracer sparsity patterns,
including second-order interactions to update the `hessian` field of `HessianTracer`.

This is functionally equivalent to:
```julia
reduce(^, tracers)
```
"""
function second_order_or(ts::AbstractArray{T}) where {T<:AbstractTracer}
    # TODO: improve performance
    return reduce(second_order_or, ts; init=myempty(T))
end

function second_order_or(a::T, b::T) where {T<:ConnectivityTracer}
    return connectivity_tracer_2_to_1(a, b, false, false)
end
function second_order_or(a::T, b::T) where {T<:GradientTracer}
    return gradient_tracer_2_to_1(a, b, false, false)
end
function second_order_or(a::T, b::T) where {T<:HessianTracer}
    return hessian_tracer_2_to_1(a, b, false, false, false, false, false)
end

"""
    first_order_or(tracers)

Compute the most conservative elementwise OR of tracer sparsity patterns,
excluding second-order interactions of `HessianTracer`.

This is functionally equivalent to:
```julia
reduce(+, tracers)
```
"""
function first_order_or(ts::AbstractArray{T}) where {T<:AbstractTracer}
    # TODO: improve performance
    return reduce(first_order_or, ts; init=myempty(T))
end
function first_order_or(a::T, b::T) where {T<:ConnectivityTracer}
    return connectivity_tracer_2_to_1(a, b, false, false)
end
function first_order_or(a::T, b::T) where {T<:GradientTracer}
    return gradient_tracer_2_to_1(a, b, false, false)
end
function first_order_or(a::T, b::T) where {T<:HessianTracer}
    return hessian_tracer_2_to_1(a, b, false, true, false, true, true)
end

#==================#
# LinearAlgebra.jl #
#==================#

# TODO: replace `second_order_or` by less conservative sparsity patterns when possible

## Determinant
LinearAlgebra.det(A::AbstractMatrix{T}) where {T<:AbstractTracer} = second_order_or(A)
LinearAlgebra.logdet(A::AbstractMatrix{T}) where {T<:AbstractTracer} = second_order_or(A)
LinearAlgebra.logabsdet(A::AbstractMatrix{T}) where {T<:AbstractTracer} = second_order_or(A)

## Norm
function LinearAlgebra.norm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return second_order_or(A)
end
function LinearAlgebra.opnorm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return second_order_or(A)
end
function LinearAlgebra.opnorm(A::AbstractMatrix{T}, p::Real=2) where {T<:AbstractTracer}
    return second_order_or(A)
end

## Eigenvalues

function LinearAlgebra.eigmax(
    A::Union{T,AbstractMatrix{T}}; permute::Bool=true, scale::Bool=true
) where {T<:AbstractTracer}
    return second_order_or(A)
end
function LinearAlgebra.eigmin(
    A::Union{T,AbstractMatrix{T}}; permute::Bool=true, scale::Bool=true
) where {T<:AbstractTracer}
    return second_order_or(A)
end
function LinearAlgebra.eigen(
    A::AbstractMatrix{T};
    permute::Bool=true,
    scale::Bool=true,
    sortby::Union{Function,Nothing}=nothing,
) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    t = second_order_or(A)
    values = Fill(t, n)
    vectors = Fill(t, n, n)
    return LinearAlgebra.Eigen(values, vectors)
end

## Inverse
function LinearAlgebra.inv(A::StridedMatrix{T}) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    t = second_order_or(A)
    return Fill(t, size(A)...)
end
function LinearAlgebra.pinv(
    A::AbstractMatrix{T}; atol::Real=0.0, rtol::Real=0.0
) where {T<:AbstractTracer}
    n, m = size(A)
    t = second_order_or(A)
    return Fill(t, m, n)
end

## Division
function LinearAlgebra.:\(
    A::AbstractMatrix{T}, B::AbstractVecOrMat
) where {T<:AbstractTracer}
    Ainv = LinearAlgebra.pinv(A)
    return Ainv * B
end

## Exponent
function LinearAlgebra.exp(A::AbstractMatrix{T}) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    t = second_order_or(A)
    return Fill(t, n, n)
end

## Matrix power
function LinearAlgebra.:^(A::AbstractMatrix{T}, p::Integer) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    t = second_order_or(A)
    return Fill(t, n, n)
end
