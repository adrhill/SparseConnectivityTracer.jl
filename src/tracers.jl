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

# TODO: refactor this properly!
sparse_matrix(T, index) = T([index;;])
sparse_matrix(::Type{Dict{I,S}}, index) where {I,S} = Dict(convert(I, index) => S())

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

empty(::Type{ConnectivityTracer{C}}) where {C} = ConnectivityTracer{C}(C())

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{C}(::Number) where {C} = empty(ConnectivityTracer{C})
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
function uniontracer(a::ConnectivityTracer{C}, b::ConnectivityTracer{C}) where {C}
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

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
    grad::G # sparse binary vector representing non-zero entries in the gradient
end

function Base.show(io::IO, t::GlobalGradientTracer{G}) where {G}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "GlobalGradientTracer{$G}(", ',', ')', true
    )
end

empty(::Type{GlobalGradientTracer{G}}) where {G} = GlobalGradientTracer{G}(G())

GlobalGradientTracer{G}(::Number) where {G} = empty(GlobalGradientTracer{G})
GlobalGradientTracer(t::GlobalGradientTracer) = t

## Unions of tracers
function uniontracer(a::GlobalGradientTracer{G}, b::GlobalGradientTracer{G}) where {G}
    return GlobalGradientTracer(union(a.grad, b.grad))
end

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
    grad::G # sparse binary vector representation of non-zero entries in the gradient
    hessian::H  # sparse binary matrix representation non-zero entries in the Hessian
end
function Base.show(io::IO, t::GlobalHessianTracer{G,H}) where {G,H}
    println(io, "$(eltype(t))(")
    for key in keys(t.hessian)
        print(io, "  ", Int(key), " => ")
        Base.show_delim_array(io, convert.(Int, t.hessian[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

empty(::Type{GlobalHessianTracer{G,H}}) where {G,H} = GlobalHessianTracer{G,H}(G(), H())

GlobalHessianTracer{G,H}(::Number) where {G,H} = empty(GlobalHessianTracer{G,H})
GlobalHessianTracer(t::GlobalHessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::GlobalHessianTracer{G,H}) where {G,H}
    d = deepcopy(t.hessian)
    s = keys2set(G, d)
    for (k, v) in pairs(d)
        d[k] = union(v, s)  # ignores symmetry
    end
    return GlobalHessianTracer{G,H}(empty_sparse_vector(G), d)
end

# Merge first- and second-order terms in an "additive" fashion
function additive_merge(
    a::GlobalHessianTracer{G,H}, b::GlobalHessianTracer{G,H}
) where {G,H}
    return GlobalHessianTracer{G,H}(
        empty_sparse_vector(G), mergewith(union, a.hessian, b.hessian)
    )
end

# Merge first- and second-order terms in a "distributive" fashion
function distributive_merge(
    a::GlobalHessianTracer{G,H}, b::GlobalHessianTracer{G,H}
) where {G,H}
    da = deepcopy(a.hessian)
    db = deepcopy(b.hessian)
    sa = keys2set(G, da)
    sb = keys2set(G, db)

    # add second-order interaction term by ignoring symmetry
    for (ka, va) in pairs(da)
        da[ka] = union(va, sb)
    end
    for (kb, vb) in pairs(db)
        db[kb] = union(vb, sa)
    end
    return GlobalHessianTracer{G,H}(empty_sparse_vector(G), merge(da, db))
end

#===========#
# Utilities #
#===========#

## Access inputs
"""
    inputs(tracer)

Return input indices of a [`ConnectivityTracer`](@ref) or [`GlobalGradientTracer`](@ref)
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::GlobalGradientTracer) = collect(t.grad)
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
    # TODO: refactor to
    # return GlobalHessianTracer{G,H}(sparse_vector(G, index), empty_sparse_matrix(H))
    return GlobalHessianTracer{G,H}(empty_sparse_vector(G), sparse_matrix(H, index))
end
