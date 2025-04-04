#===========#
# Utilities #
#===========#

function split_dual_array(A::AbstractArray{D}) where {D<:Dual}
    primals = getproperty.(A, :primal)
    tracers = getproperty.(A, :tracer)
    return primals, tracers
end

sct_owns_type(::Type) = false
sct_owns_type(::Type{T}) where {T<:AbstractTracer} = true
sct_owns_type(::Type{A}) where {A<:AbstractArray{<:AbstractTracer}} = true
sct_owns_type(::Type{A}) where {T,A<:AbstractArray{T}} = sct_owns_type(T)

nopiracy(types) = any(sct_owns_type, types)

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

## Norm
function LinearAlgebra.norm(A::AbstractArray{T}, p::Real=2) where {T<:AbstractTracer}
    if isone(p) || isinf(p)
        return first_order_or(A)
    else
        return second_order_or(A)
    end
end
function LinearAlgebra.opnorm(A::AbstractMatrix{T}, p::Real=2) where {T<:AbstractTracer}
    if isone(p) || isinf(p)
        return first_order_or(A)
    else
        return second_order_or(A)
    end
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
function Base.inv(A::StridedMatrix{T}) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    t = second_order_or(A)
    return Fill(t, size(A)...)
end
function Base.inv(D::Diagonal{T}) where {T<:AbstractTracer}
    ts_in = D.diag
    ts_out = similar(ts_in)
    for i in 1:length(ts_out)
        ts_out[i] = inv(ts_in[i])
    end
    return Diagonal(ts_out)
end

function LinearAlgebra.pinv(
    A::AbstractMatrix{T}; atol::Real=0.0, rtol::Real=0.0
) where {T<:AbstractTracer}
    n, m = size(A)
    t = second_order_or(A)
    return Fill(t, m, n)
end
LinearAlgebra.pinv(D::Diagonal{T}) where {T<:AbstractTracer} = inv(D)

## Dot product – adapted from https://github.com/JuliaLang/LinearAlgebra.jl/blob/924dda4d5d26d745fc8993b7ffdfaa80ee0e0c0e/src/generic.jl#L895-L1029
LinearAlgebra.dot(x::T, y::T) where {T<:AbstractTracer} = x * y # no conjugate required on tracers.

# In the future, we will likely have to add more methods.
for (Tx, TA, Ty) in Iterators.filter(
    nopiracy, # only keep tuples of types we own 
    Iterators.product(
        # Types for x
        (Vector, Vector{<:AbstractTracer}, SubArray, SubArray{<:AbstractTracer,1}),
        # Types for A
        (Matrix, Matrix{<:AbstractTracer}),
        # Types for y
        (Vector, Vector{<:AbstractTracer}, SubArray, SubArray{<:AbstractTracer,1}),
    ),
)
    @eval LinearAlgebra.dot(x::$Tx, A::$TA, y::$Ty) = LinearAlgebra.dot(x, A * y)
end

## Division
function LinearAlgebra.:\(
    A::AbstractMatrix{T}, B::AbstractVecOrMat
) where {T<:AbstractTracer}
    Ainv = LinearAlgebra.pinv(A)
    return Ainv * B
end

## Exponential
function Base.exp(A::AbstractMatrix{T}) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    t = second_order_or(A)
    return Fill(t, n, n)
end

## Matrix power
function LinearAlgebra.:^(A::AbstractMatrix{T}, p::Integer) where {T<:AbstractTracer}
    LinearAlgebra.checksquare(A)
    n = size(A, 1)
    if iszero(p)
        return Fill(myempty(T), n, n)
    else
        t = second_order_or(A)
        return Fill(t, n, n)
    end
end

function Base.literal_pow(::typeof(^), D::Diagonal{T}, ::Val{0}) where {T<:AbstractTracer}
    ts = similar(D.diag)
    ts .= myempty(T)
    return Diagonal(ts)
end

## clamp!
Base.clamp!(A::AbstractArray{T}, lo, hi) where {T<:AbstractTracer} = A
function Base.clamp!(A::AbstractArray{T}, lo::T, hi) where {T<:AbstractTracer}
    return first_order_or.(A, lo)
end
function Base.clamp!(A::AbstractArray{T}, lo, hi::T) where {T<:AbstractTracer}
    return first_order_or.(A, hi)
end
function Base.clamp!(A::AbstractArray{T}, lo::T, hi::T) where {T<:AbstractTracer}
    return first_order_or.(A, first_order_or(lo, hi))
end

#==========================#
# LinearAlgebra.jl on Dual #
#==========================#

# `Duals` should use LinearAlgebra's generic fallback implementations
# to compute the "least conservative" sparsity patterns possible on a scalar level.

# The following three methods are a temporary fix for issue #108.
# TODO: instead overload `lu` on AbstractMatrix of Duals.
function LinearAlgebra.det(A::AbstractMatrix{D}) where {D<:Dual}
    primals, tracers = split_dual_array(A)
    p = LinearAlgebra.logdet(primals)
    t = LinearAlgebra.logdet(tracers)
    return D(p, t)
end
function LinearAlgebra.logdet(A::AbstractMatrix{D}) where {D<:Dual}
    primals, tracers = split_dual_array(A)
    p = LinearAlgebra.logdet(primals)
    t = LinearAlgebra.logdet(tracers)
    return D(p, t)
end
function LinearAlgebra.logabsdet(A::AbstractMatrix{D}) where {D<:Dual}
    primals, tracers = split_dual_array(A)
    p1, p2 = LinearAlgebra.logabsdet(primals)
    t1, t2 = LinearAlgebra.logabsdet(tracers)
    return (D(p1, t1), D(p2, t2))
end
function LinearAlgebra.:\(A::AbstractMatrix{<:Dual}, B::AbstractVector)
    primals, tracers = split_dual_array(A)
    p = primals \ B
    t = tracers \ B
    return Dual.(p, t)
end
function LinearAlgebra.:\(A::AbstractMatrix, B::AbstractVector{D}) where {D<:Dual}
    return D.(A) \ B
end
function LinearAlgebra.:\(
    A::AbstractMatrix{D1}, B::AbstractVector{D2}
) where {D1<:Dual,D2<:Dual}
    A_primals, A_tracers = split_dual_array(A)
    B_primals, B_tracers = split_dual_array(B)
    p = A_primals \ B_primals
    t = A_tracers \ B_tracers
    return Dual.(p, t)
end

#==============#
# SparseArrays #
#==============#

# Helper function needed in SparseArrays's sparsematrix, sparsevector and higherorderfns.
# On Tracers, `iszero` and `!iszero` don't return a boolean, 
# but we need a function that does to handle the structure of the array.

SparseArrays._iszero(t::AbstractTracer) = isemptytracer(t)
SparseArrays._iszero(d::Dual) = isemptytracer(tracer(d)) && iszero(primal(d))

SparseArrays._isnotzero(t::AbstractTracer) = !isemptytracer(t)
SparseArrays._isnotzero(d::Dual) = !isemptytracer(tracer(d)) || !iszero(primal(d))
