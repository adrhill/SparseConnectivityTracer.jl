#= This file handles the actual tracing of functions:
1) creating tracers from inputs
2) evaluating the function with the created tracers

The resulting output is parsed in `src/parse_outputs_to_matrix.jl`.
=#

#==================#
# Enumerate inputs #
#==================#

"""
    trace_input(T, x)
    trace_input(T, xs)

Enumerates input indices and constructs the specified type `T` of tracer.
Supports [`GradientTracer`](@ref), [`HessianTracer`](@ref) and [`Dual`](@ref).
"""
function trace_input(::Type{T}, xs) where {T <: Union{AbstractTracer, Dual}}
    return _trace_input(T, xs, 1, length(xs))
end

# If possible, this should call `similar` and have the function signature `A{T} -> A{Int}`.
# For some array types, this function signature isn't possible,
# e.g. on `Symmetric`, where symmetry doesn't hold for the index matrix.
allocate_index_matrix(A::AbstractArray) = similar(A, Int)
allocate_index_matrix(A::Symmetric) = Matrix{Int}(undef, size(A)...)

function _trace_input(
        ::Type{T}, xs::AbstractArray, i::Integer, j::Integer
    ) where {T <: Union{AbstractTracer, Dual}}
    inds = allocate_index_matrix(xs)
    inds .= reshape(1:length(xs), size(xs))
    return create_tracers(T, xs, inds, i, j)
end

function _trace_input(
        ::Type{T}, xs::Diagonal, i::Integer, j::Integer
    ) where {T <: Union{AbstractTracer, Dual}}
    ts = create_tracers(T, diag(xs), diagind(xs), i, j)
    return Diagonal(ts)
end

function _trace_input(
        ::Type{T}, x::Real, i::Integer, j::Integer
    ) where {T <: Union{AbstractTracer, Dual}}
    return only(_trace_input(T, [x], i, j))
end

#=========================#
# Trace through functions #
#=========================#

function trace_function(
        ::Type{T}, f, x, i::Integer, j::Integer
    ) where {T <: Union{AbstractTracer, Dual}}
    xt = _trace_input(T, x, i, j)
    yt = f(xt)
    return xt, yt
end

function trace_function(
        ::Type{T}, f!, y, x, i::Integer, j::Integer
    ) where {T <: AbstractTracer}
    xt = _trace_input(T, x, i, j)
    yt = similar(y, T)
    fill!(yt, myempty(T))
    f!(yt, xt)
    return xt, yt
end

function trace_function(
        ::Type{D}, f!, y, x, i::Integer, j::Integer
    ) where {P, T <: AbstractTracer, D <: Dual{P, T}}
    t = myempty(T)
    xt = _trace_input(D, x, i, j)
    yt = Dual.(y, t)
    f!(yt, xt)
    return xt, yt
end

to_array(x::Real) = [x]
to_array(x::AbstractArray) = x

#==========#
# Jacobian #
#==========#

# Compute the sparsity pattern of the Jacobian of `y = f(x)`.
function _jacobian_sparsity(
        f, x, ::Type{T} = DEFAULT_GRADIENT_TRACER
    ) where {T <: GradientTracer}
    return maximum(chunks(T, x)) do interval
        i, j = first(interval), last(interval)
        xt, yt = trace_function(T, f, x, i, j)
        return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
    end
end

# Compute the sparsity pattern of the Jacobian of `f!(y, x)`.
function _jacobian_sparsity(
        f!, y, x, ::Type{T} = DEFAULT_GRADIENT_TRACER
    ) where {T <: GradientTracer}
    return maximum(chunks(T, x)) do interval
        i, j = first(interval), last(interval)
        xt, yt = trace_function(T, f!, y, x, i, j)
        return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
    end
end

# Compute the local sparsity pattern of the Jacobian of `y = f(x)` at `x`.
function _local_jacobian_sparsity(
        f, x, ::Type{T} = DEFAULT_GRADIENT_TRACER
    ) where {T <: GradientTracer}
    D = Dual{eltype(x), T}
    return maximum(chunks(D, x)) do interval
        i, j = first(interval), last(interval)
        xt, yt = trace_function(D, f, x, i, j)
        return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
    end
end

# Compute the local sparsity pattern of the Jacobian of `f!(y, x)` at `x`.
function _local_jacobian_sparsity(
        f!, y, x, ::Type{T} = DEFAULT_GRADIENT_TRACER
    ) where {T <: GradientTracer}
    D = Dual{eltype(x), T}
    return maximum(chunks(D, x)) do interval
        i, j = first(interval), last(interval)
        xt, yt = trace_function(D, f!, y, x, i, j)
        return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
    end
end

#=========#
# Hessian #
#=========#

# Compute the sparsity pattern of the Hessian of a scalar function `y = f(x)`.
function _hessian_sparsity(f, x, ::Type{T} = DEFAULT_HESSIAN_TRACER) where {T <: HessianTracer}
    xt, yt = trace_function(T, f, x, 1, length(x))
    return hessian_tracers_to_matrix(to_array(xt), yt)
end

# Compute the local sparsity pattern of the Hessian of a scalar function `y = f(x)` at `x`.
function _local_hessian_sparsity(
        f, x, ::Type{T} = DEFAULT_HESSIAN_TRACER
    ) where {T <: HessianTracer}
    D = Dual{eltype(x), T}
    xt, yt = trace_function(D, f, x, 1, length(x))
    return hessian_tracers_to_matrix(to_array(xt), yt)
end
