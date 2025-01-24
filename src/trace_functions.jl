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
trace_input(::Type{T}, xs) where {T<:Union{AbstractTracer,Dual}} = trace_input(T, xs, 1)

# If possible, this should call `similar` and have the function signature `A{T} -> A{Int}`.
# For some array types, this function signature isn't possible, 
# e.g. on `Symmetric`, where symmetry doesn't hold for the index matrix.
allocate_index_matrix(A::AbstractArray) = similar(A, Int)
allocate_index_matrix(A::Symmetric) = Matrix{Int}(undef, size(A)...)

function trace_input(::Type{T}, xs::AbstractArray, i) where {T<:Union{AbstractTracer,Dual}}
    is = allocate_index_matrix(xs)
    is .= reshape(1:length(xs), size(xs)) .+ (i - 1)
    return create_tracers(T, xs, is)
end

function trace_input(::Type{T}, xs::Diagonal, i) where {T<:Union{AbstractTracer,Dual}}
    ts = create_tracers(T, diag(xs), diagind(xs))
    return Diagonal(ts)
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

function trace_function(::Type{T}, f!, y, x) where {T<:AbstractTracer}
    xt = trace_input(T, x)
    yt = fill(myempty(T), size(y))
    f!(yt, xt)
    return xt, yt
end

function trace_function(::Type{D}, f!, y, x) where {P,T<:AbstractTracer,D<:Dual{P,T}}
    t = myempty(T)
    xt = trace_input(D, x)
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
    f, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    xt, yt = trace_function(T, f, x)
    return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
end

# Compute the sparsity pattern of the Jacobian of `f!(y, x)`.
function _jacobian_sparsity(
    f!, y, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    xt, yt = trace_function(T, f!, y, x)
    return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
end

# Compute the local sparsity pattern of the Jacobian of `y = f(x)` at `x`.
function _local_jacobian_sparsity(
    f, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f, x)
    return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
end

# Compute the local sparsity pattern of the Jacobian of `f!(y, x)` at `x`.
function _local_jacobian_sparsity(
    f!, y, x, ::Type{T}=DEFAULT_GRADIENT_TRACER
) where {T<:GradientTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f!, y, x)
    return jacobian_tracers_to_matrix(to_array(xt), to_array(yt))
end

#=========#
# Hessian #
#=========#

# Compute the sparsity pattern of the Hessian of a scalar function `y = f(x)`.
function _hessian_sparsity(f, x, ::Type{T}=DEFAULT_HESSIAN_TRACER) where {T<:HessianTracer}
    xt, yt = trace_function(T, f, x)
    return hessian_tracers_to_matrix(to_array(xt), yt)
end

# Compute the local sparsity pattern of the Hessian of a scalar function `y = f(x)` at `x`.
function _local_hessian_sparsity(
    f, x, ::Type{T}=DEFAULT_HESSIAN_TRACER
) where {T<:HessianTracer}
    D = Dual{eltype(x),T}
    xt, yt = trace_function(D, f, x)
    return hessian_tracers_to_matrix(to_array(xt), yt)
end
