abstract type AbstractTracer <: Number end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)

const SET_TYPE_MESSAGE = """
The provided index set type `S` has to satisfy the following conditions:

- it is an iterable with `<:Integer` element type
- it implements `union`

Subtypes of `AbstractSet{<:Integer}` are a natural choice, like `BitSet` or `Set{UInt64}`.
"""

empty_sparse_vector(T) = T()
empty_sparse_matrix(T) = T()

sparse_vector(T, index) = T([index])
sparse_vector(::Type{T}, index) where {T<:DuplicateVector} = T(index)
sparse_vector(::Type{T}, index) where {T<:SortedVector} = T(index)

#================#
# Elementwise OR #
#================#

## We use ∨ to represent the elementwise OR
# REVIEW TODO: \vee looks to much like the character 'v' in code
# Gradient representations
∨(a::G, b::G) where {G<:AbstractSet} = a ∪ b
∨(a::G, b::G) where {G<:DuplicateVector} = G(vcat(a.data, b.data))
∨(a::G, b::G) where {G<:SortedVector} = a ∪ b
∨(a::G, b::G) where {G<:RecursiveSet} = a ∪ b

# Hessian representations
# REVIEW TODO: for now, we assume Hessians are represented as a set of tuples
∨(a::H, b::H) where {I<:Integer,H<:AbstractSet{Tuple{I,I}}} = a ∪ b

## Outer product on gradients
# Compute `out ∨ (𝟙[∇a] ∨ 𝟙[∇b]ᵀ)` in out
# TODO: add special dispatches based on type of G
function outer_product_or!(
    out::H, a::G, b::G
) where {I<:Integer,H<:AbstractSet{Tuple{I,I}},G}
    for i in a
        for j in b
            push!(out, (i, j))
        end
    end
    return out
end

#==============#
# Connectivity #
#==============#

"""
    ConnectivityTracer{C}(indexset) <: Number

Number type keeping track of input indices of previous computations.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`connectivity_pattern`](@ref).
"""
struct ConnectivityTracer{C} <: AbstractTracer
    inputs::C # sparse binary vector representing non-zero indices of connected, enumerated inputs
end

function Base.show(io::IO, t::ConnectivityTracer{C}) where {C}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "ConnectivityTracer{$C}(", ',', ')', true
    )
end

function empty(::Type{ConnectivityTracer{C}}) where {C}
    return ConnectivityTracer{C}(empty_sparse_vector(C))
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{C}(::Number) where {C} = empty(ConnectivityTracer{C})
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
∨(a::T, b::T) where {T<:ConnectivityTracer} = T(a.inputs ∨ b.inputs)

#==========#
# Jacobian #
#==========#

"""
    GlobalGradientTracer{G}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`jacobian_pattern`](@ref).
"""
struct GlobalGradientTracer{G} <: AbstractTracer
    gradient::G # sparse binary vector representing non-zero entries in the gradient
end

function Base.show(io::IO, t::GlobalGradientTracer{G}) where {G}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "GlobalGradientTracer{$G}(", ',', ')', true
    )
end

function empty(::Type{GlobalGradientTracer{G}}) where {G}
    return GlobalGradientTracer{G}(empty_sparse_vector(G))
end

GlobalGradientTracer{G}(::Number) where {G} = empty(GlobalGradientTracer{G})
GlobalGradientTracer(t::GlobalGradientTracer) = t

## Unions of tracers
∨(a::T, b::T) where {T<:GlobalGradientTracer} = T(a.gradient ∨ b.gradient)

#=========#
# Hessian #
#=========#
"""
    GlobalHessianTracer{G,H}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`hessian_pattern`](@ref).
"""
struct GlobalHessianTracer{G,H} <: AbstractTracer
    gradient::G # sparse binary vector representation of non-zero entries in the gradient
    hessian::H  # sparse binary matrix representation of non-zero entries in the Hessian
end
function Base.show(io::IO, t::GlobalHessianTracer{G,H}) where {G,H}
    println(io, "$(eltype(t))(")
    println(io, "  Gradient: ", t.gradient, ",")
    println(io, "  Hessian:  ", t.hessian)
    print(io, ")")
    return nothing
end

function empty(::Type{GlobalHessianTracer{G,H}}) where {G,H}
    return GlobalHessianTracer{G,H}(empty_sparse_vector(G), empty_sparse_matrix(H))
end

GlobalHessianTracer{G,H}(::Number) where {G,H} = empty(GlobalHessianTracer{G,H})
GlobalHessianTracer(t::GlobalHessianTracer) = t

#===========#
# Utilities #
#===========#

## Access inputs
"""
    inputs(tracer)

Return input indices of a [`ConnectivityTracer`](@ref) or [`GlobalGradientTracer`](@ref)
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::GlobalGradientTracer) = collect(t.gradient)
inputs(t::GlobalHessianTracer, i::Integer) = collect(t.hessian[i])

"""
    tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GlobalGradientTracer`](@ref) and [`GlobalHessianTracer`](@ref) from input indices.
"""
function tracer(::Type{GlobalGradientTracer{G}}, index::Integer) where {G}
    return GlobalGradientTracer{G}(sparse_vector(G, index))
end
function tracer(::Type{ConnectivityTracer{C}}, index::Integer) where {C}
    return ConnectivityTracer{C}(sparse_vector(C, index))
end
function tracer(::Type{GlobalHessianTracer{G,H}}, index::Integer) where {G,H}
    return GlobalHessianTracer{G,H}(sparse_vector(G, index), empty_sparse_matrix(H))
end
