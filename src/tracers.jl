abstract type AbstractTracer <: Number end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)
empty(T) = T()

sparse_vector(T, index) = T([index])

#================#
# Elementwise OR #
#================#

×(a::G, b::G) where {G<:AbstractSet} = Set((i, j) for i in a, j in b)

#==============#
# Connectivity #
#==============#

"""
$(TYPEDEF)

`Number` type keeping track of input indices of previous computations.

For a higher-level interface, refer to [`connectivity_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> inputs = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> SparseConnectivityTracer.ConnectivityTracer(inputs)
SparseConnectivityTracer.ConnectivityTracer{Set{Int64}}(1, 3)
```
"""
struct ConnectivityTracer{C<:AbstractSet{<:Integer}} <: AbstractTracer
    "Sparse binary vector representing non-zero indices of connected, enumerated inputs."
    inputs::C
end

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort!(collect(t.inputs))), "$(typeof(t))(", ',', ')', true
    )
end

function empty(::Type{ConnectivityTracer{C}}) where {C}
    return ConnectivityTracer{C}(empty(C))
end

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
function ConnectivityTracer{C}(::Number) where {C<:AbstractSet{<:Integer}}
    return empty(ConnectivityTracer{C})
end

ConnectivityTracer{C}(t::ConnectivityTracer{C}) where {C<:AbstractSet{<:Integer}} = t
ConnectivityTracer(t::ConnectivityTracer) = t

#=================#
# Gradient Tracer #
#=================#

"""
$(TYPEDEF)

`Number` type keeping track of non-zero gradient entries.

For a higher-level interface, refer to [`jacobian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> grad = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> SparseConnectivityTracer.GlobalGradientTracer(grad)
SparseConnectivityTracer.GlobalGradientTracer{Set{Int64}}(1, 3)
```
"""
struct GlobalGradientTracer{G<:AbstractSet{<:Integer}} <: AbstractTracer
    "Sparse binary vector representing non-zero entries in the gradient."
    grad::G
end

function Base.show(io::IO, t::GlobalGradientTracer)
    return Base.show_delim_array(
        io, convert.(Int, sort(collect(t.grad))), "$(typeof(t))(", ',', ')', true
    )
end

function empty(::Type{GlobalGradientTracer{G}}) where {G}
    return GlobalGradientTracer{G}(empty(G))
end

function GlobalGradientTracer{G}(::Number) where {G<:AbstractSet{<:Integer}}
    return empty(GlobalGradientTracer{G})
end

GlobalGradientTracer{G}(t::GlobalGradientTracer{G}) where {G<:AbstractSet{<:Integer}} = t
GlobalGradientTracer(t::GlobalGradientTracer) = t

#=========#
# Hessian #
#=========#

"""
$(TYPEDEF)

`Number` type keeping track of non-zero gradient and Hessian entries.

For a higher-level interface, refer to [`hessian_pattern`](@ref).

## Fields
$(TYPEDFIELDS)

## Example
```jldoctest
julia> grad = Set([1, 3])
Set{Int64} with 2 elements:
  3
  1

julia> hess = Set([(1, 1), (2, 3), (3, 2)])
Set{Tuple{Int64, Int64}} with 3 elements:
  (3, 2)
  (1, 1)
  (2, 3)

julia> SparseConnectivityTracer.GlobalHessianTracer(grad, hess)
SparseConnectivityTracer.GlobalHessianTracer{Set{Int64}, Set{Tuple{Int64, Int64}}}(
  Gradient: Set([3, 1]),
  Hessian:  Set([(3, 2), (1, 1), (2, 3)])
)
```
"""
struct GlobalHessianTracer{
    G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}
} <: AbstractTracer
    "Sparse binary vector representation of non-zero entries in the gradient."
    grad::G
    "Sparse binary matrix representation of non-zero entries in the Hessian."
    hess::H
end

function Base.show(io::IO, t::GlobalHessianTracer)
    println(io, "$(eltype(t))(")
    println(io, "  Gradient: ", t.grad, ",")
    println(io, "  Hessian:  ", t.hess)
    print(io, ")")
    return nothing
end

function empty(::Type{GlobalHessianTracer{G,H}}) where {G,H}
    return GlobalHessianTracer{G,H}(empty(G), empty(H))
end

function GlobalHessianTracer{G,H}(
    ::Number
) where {G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}}
    return empty(GlobalHessianTracer{G,H})
end

function GlobalHessianTracer{G,H}(
    t::GlobalHessianTracer{G,H}
) where {G<:AbstractSet{<:Integer},H<:AbstractSet{<:Tuple{Integer,Integer}}}
    return t
end
GlobalHessianTracer(t::GlobalHessianTracer) = t

#===========#
# Utilities #
#===========#

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
    return GlobalHessianTracer{G,H}(sparse_vector(G, index), empty(H))
end
