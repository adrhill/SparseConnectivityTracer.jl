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

#===========#
# Utilities #
#===========#

function split_dual_array(A::AbstractArray{D}) where {D<:Dual}
    primals = getproperty.(A, :primal)
    tracers = getproperty.(A, :tracer)
    return primals, tracers
end
function split_dual_array(A::SparseArrays.SparseMatrixCSC{D}) where {D<:Dual}
    A = Matrix(A)
    primals = getproperty.(A, :primal)
    tracers = getproperty.(A, :tracer)
    return sparse(primals), sparse(tracers)
end

#==================#
# LinearAlgebra.jl #
#==================#

# TODO: replace `second_order_or` by less conservative sparsity patterns when possible

## Determinant
LinearAlgebra.det(A::AbstractMatrix{T}) where {T<:AbstractTracer} = second_order_or(A)
LinearAlgebra.logdet(A::AbstractMatrix{T}) where {T<:AbstractTracer} = second_order_or(A)
function LinearAlgebra.logabsdet(A::AbstractMatrix{T}) where {T<:AbstractTracer}
    t1 = second_order_or(A)
    t2 = sign(t1) # corresponds to sign of det(A): set first- and second-order derivatives to zero
    return (t1, t2)
end

# Fix for issue #108
function LinearAlgebra.det(A::AbstractMatrix{D}) where {D<:Dual}
    P, T = split_dual_array(A)
    p = LinearAlgebra.logdet(P)
    t = LinearAlgebra.logdet(T)
    return D(p, t)
end
function LinearAlgebra.logdet(A::AbstractMatrix{D}) where {D<:Dual}
    P, T = split_dual_array(A)
    p = LinearAlgebra.logdet(P)
    t = LinearAlgebra.logdet(T)
    return D(p, t)
end
function LinearAlgebra.logabsdet(A::AbstractMatrix{D}) where {D<:Dual}
    P, T = split_dual_array(A)
    p1, p2 = LinearAlgebra.logabsdet(P)
    t1, t2 = LinearAlgebra.logabsdet(T)
    return (D(p1, t1), D(p2, t2))
end

## Norm
function LinearAlgebra.norm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return second_order_or(A)
end
function LinearAlgebra.opnorm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    return first_order_or(A)
end
function LinearAlgebra.opnorm(A::AbstractMatrix{T}, p::Real=2) where {T<:AbstractTracer}
    return first_order_or(A)
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

#==============#
# SparseArrays #
#==============#

# Conversion of matrices of tracers to SparseMatrixCSC has to be rewritten 
# due to use of `count(_isnotzero, M)` in SparseArrays.jl
#
# Code modified from MIT licensed SparseArrays.jl source:
# https://github.com/JuliaSparse/SparseArrays.jl/blob/45dfe459ede2fa1419e7068d4bda92d9d22bd44d/src/sparsematrix.jl#L901-L920
# Copyright (c) 2009-2024: Jeff Bezanson, Stefan Karpinski, Viral B. Shah, and other contributors: https://github.com/JuliaLang/julia/contributors
function SparseArrays.SparseMatrixCSC{Tv,Ti}(
    M::StridedMatrix{Tv}
) where {Tv<:AbstractTracer,Ti}
    nz = count(!isemptytracer, M)
    colptr = zeros(Ti, size(M, 2) + 1)
    nzval = Vector{Tv}(undef, nz)
    rowval = Vector{Ti}(undef, nz)
    colptr[1] = 1
    cnt = 1
    @inbounds for j in 1:size(M, 2)
        for i in 1:size(M, 1)
            v = M[i, j]
            if !isemptytracer(v)
                rowval[cnt] = i
                nzval[cnt] = v
                cnt += 1
            end
        end
        colptr[j + 1] = cnt
    end
    return SparseArrays.SparseMatrixCSC(size(M, 1), size(M, 2), colptr, rowval, nzval)
end
