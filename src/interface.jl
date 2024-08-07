const DEFAULT_GRADIENT_TRACER = GradientTracer{IndexSetGradientPattern{Int,BitSet}}
const DEFAULT_HESSIAN_TRACER = HessianTracer{
    IndexSetHessianPattern{Int,BitSet,Set{Tuple{Int,Int}},false}
}

#==================#
# Enumerate inputs #
#==================#

"""
    trace_input(T, x)
    trace_input(T, xs)

Enumerates input indices and constructs the specified type `T` of tracer.
Supports [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref).
"""
trace_input(::Type{T}, xs) where {T<:Union{AbstractTracer,Dual}} = trace_input(T, xs, 1)

# If possible, this should call `similar` and have a function signature `A -> A`.
# For some array types like `Symmetric`, this function signature isn't possible, 
# e.g. because symmetry doesn't hold for the index matrix.
allocate_index_matrix(A::AbstractArray) = similar(A, Int)
allocate_index_matrix(A::Symmetric) = Matrix{Int}(undef, size(A)...)
allocate_index_matrix(A::Diagonal) = Matrix{Int}(undef, size(A)...)

function trace_input(::Type{T}, xs::AbstractArray, i) where {T<:Union{AbstractTracer,Dual}}
    is = allocate_index_matrix(xs)
    is .= reshape(1:length(xs), size(xs)) .+ (i - 1)
    return create_tracers(T, xs, is)
end

function trace_input(::Type{T}, x::Real, i::Integer) where {T<:Union{AbstractTracer,Dual}}
    return only(create_tracers(T, [x], [i]))
end

#=========================#
# Trace through functions #
#=========================#

function trace_function(::Type{T}, f, x) where {T<:Union{AbstractTracer,Dual}}
    xt = trace_input(T, x)
    yt = f(xt)
    return xt, yt
end

function trace_function(::Type{T}, f!, y, x) where {T<:Union{AbstractTracer,Dual}}
    xt = trace_input(T, x)
    yt = similar(y, T)
    f!(yt, xt)
    return xt, yt
end

to_array(x::Real) = [x]
to_array(x::AbstractArray) = x

# Utilities
_tracer_or_number(x::Real) = x
_tracer_or_number(d::Dual) = tracer(d)

#================#
# GradientTracer #
#================#

# Compute the sparsity pattern of the Jacobian of `y = f(x)`.
function _jacobian_sparsity(
    f, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    xt, yt = trace_function(T, f, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

# Compute the sparsity pattern of the Jacobian of `f!(y, x)`.
function _jacobian_sparsity(
    f!, y, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    xt, yt = trace_function(T, f!, y, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

# Compute the local sparsity pattern of the Jacobian of `y = f(x)` at `x`.
function _local_jacobian_sparsity(
    f, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

# Compute the local sparsity pattern of the Jacobian of `f!(y, x)` at `x`.
function _local_jacobian_sparsity(
    f!, y, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f!, y, x)
    return jacobian_pattern_to_mat(to_array(xt), to_array(yt))
end

function jacobian_pattern_to_mat(
    xt::AbstractArray{T}, yt::AbstractArray{<:Real}
) where {T<:GradientTracer}
    n, m = length(xt), length(yt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values
    for (i, y) in enumerate(yt)
        if y isa T && !isemptytracer(y)
            for j in gradient(y)
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V, m, n)
end

function jacobian_pattern_to_mat(
    xt::AbstractArray{D}, yt::AbstractArray{<:Real}
) where {P,T<:GradientTracer,D<:Dual{P,T}}
    return jacobian_pattern_to_mat(tracer.(xt), _tracer_or_number.(yt))
end

#===============#
# HessianTracer #
#===============#

# Compute the sparsity pattern of the Hessian of a scalar function `y = f(x)`.
function _hessian_sparsity(f, x, ::Type{T}=DEFAULT_HESSIAN_TRACER) where {T<:HessianTracer}
    xt, yt = trace_function(T, f, x)
    return hessian_pattern_to_mat(to_array(xt), yt)
end

# Compute the local sparsity pattern of the Hessian of a scalar function `y = f(x)` at `x`.
function _local_hessian_sparsity(
    f, x, ::Type{T}=DEFAULT_HESSIAN_TRACER
) where {T<:HessianTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f, x)
    return hessian_pattern_to_mat(to_array(xt), yt)
end

function hessian_pattern_to_mat(xt::AbstractArray{T}, yt::T) where {T<:HessianTracer}
    n = length(xt)
    I = Int[] # row indices
    J = Int[] # column indices
    V = Bool[]   # values

    if !isemptytracer(yt)
        for (i, j) in hessian(yt)
            push!(I, i)
            push!(J, j)
            push!(V, true)
        end
    end
    h = sparse(I, J, V, n, n)
    return h
end

function hessian_pattern_to_mat(
    xt::AbstractArray{D1}, yt::D2
) where {P1,P2,T<:HessianTracer,D1<:Dual{P1,T},D2<:Dual{P2,T}}
    return hessian_pattern_to_mat(tracer.(xt), tracer(yt))
end

function hessian_pattern_to_mat(xt::AbstractArray{T}, yt::Number) where {T<:HessianTracer}
    return hessian_pattern_to_mat(xt, myempty(T))
end

function hessian_pattern_to_mat(
    xt::AbstractArray{D1}, yt::Number
) where {P1,T<:HessianTracer,D1<:Dual{P1,T}}
    return hessian_pattern_to_mat(tracer.(xt), myempty(T))
end
