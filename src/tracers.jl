abstract type AbstractTracer <: Number end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)

const SET_TYPE_MESSAGE = """
The provided index set type `S` has to satisfy the following conditions:

- it is an iterable with `<:Integer` element type
- it implements `union`

Subtypes of `AbstractSet{<:Integer}` are a natural choice, like `BitSet` or `Set{UInt64}`.
"""

#==============#
# Connectivity #
#==============#

"""
    ConnectivityTracer{I,S}(indexset) <: Number

Number type keeping track of input indices of previous computations.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`connectivity_pattern`](@ref).
"""
struct ConnectivityTracer{I<:Integer,S} <: AbstractTracer
    inputs::S # indices of connected, enumerated inputs
end
function ConnectivityTracer(inputs::S) where {S}
    I = eltype(S)
    return ConnectivityTracer{I,S}(inputs)
end

function Base.show(io::IO, t::ConnectivityTracer{I,S}) where {I,S}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "ConnectivityTracer{$I,$S}(", ',', ')', true
    )
end

empty(::Type{ConnectivityTracer{I,S}}) where {I<:Integer,S} = ConnectivityTracer{I,S}(S())

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{I,S}(::Number) where {I<:Integer,S} = empty(ConnectivityTracer{I,S})
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
function uniontracer(a::ConnectivityTracer{I,S}, b::ConnectivityTracer{I,S}) where {I,S}
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

#==========#
# Jacobian #
#==========#

"""
    GlobalGradientTracer{I,S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`jacobian_pattern`](@ref).
"""
struct GlobalGradientTracer{I<:Integer,S} <: AbstractTracer
    inputs::S
end
function GlobalGradientTracer(inputs::S) where {S}
    I = eltype(S)
    return GlobalGradientTracer{I,S}(inputs)
end

function Base.show(io::IO, t::GlobalGradientTracer{I,S}) where {I,S}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "GlobalGradientTracer{$I,$S}(", ',', ')', true
    )
end

function empty(::Type{GlobalGradientTracer{I,S}}) where {I<:Integer,S}
    return GlobalGradientTracer{I,S}(S())
end

GlobalGradientTracer{I,S}(::Number) where {I<:Integer,S} = empty(GlobalGradientTracer{I,S})
GlobalGradientTracer(t::GlobalGradientTracer) = t

## Unions of tracers
function uniontracer(a::GlobalGradientTracer{I,S}, b::GlobalGradientTracer{I,S}) where {I,S}
    return GlobalGradientTracer(union(a.inputs, b.inputs))
end

#=========#
# Hessian #
#=========#
"""
    GlobalHessianTracer{I,S,D}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`hessian_pattern`](@ref).
"""
struct GlobalHessianTracer{I<:Integer,S,D<:AbstractDict{I,S}} <: AbstractTracer
    inputs::D
end
function Base.show(io::IO, t::GlobalHessianTracer{I,S,D}) where {I,S,D}
    println(io, "$(eltype(t))(")
    for key in keys(t.inputs)
        print(io, "  ", Int(key), " => ")
        Base.show_delim_array(io, convert.(Int, t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

function empty(::Type{GlobalHessianTracer{I,S,D}}) where {I<:Integer,S,D<:AbstractDict{I,S}}
    return GlobalHessianTracer{I,S,D}(D())
end

function GlobalHessianTracer{I,S,D}(::Number) where {I<:Integer,S,D}
    return empty(GlobalHessianTracer{I,S,D})
end
GlobalHessianTracer(t::GlobalHessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::GlobalHessianTracer{I,S}) where {I,S}
    d = deepcopy(t.inputs)
    s = keys2set(S, d)
    for (k, v) in pairs(d)
        d[k] = union(v, s)  # ignores symmetry
    end
    return GlobalHessianTracer(d)
end

# Merge first- and second-order terms in an "additive" fashion
function additive_merge(a::GlobalHessianTracer, b::GlobalHessianTracer)
    return GlobalHessianTracer(mergewith(union, a.inputs, b.inputs))
end

# Merge first- and second-order terms in a "distributive" fashion
function distributive_merge(
    a::GlobalHessianTracer{I,S,D}, b::GlobalHessianTracer{I,S,D}
) where {I,S,D}
    da = deepcopy(a.inputs)
    db = deepcopy(b.inputs)
    sa = keys2set(S, da)
    sb = keys2set(S, db)

    # add second-order interaction term by ignoring symmetry
    for (ka, va) in pairs(da)
        da[ka] = union(va, sb)
    end
    for (kb, vb) in pairs(db)
        db[kb] = union(vb, sa)
    end
    return GlobalHessianTracer(merge(da, db))
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
inputs(t::GlobalGradientTracer) = collect(t.inputs)
inputs(t::GlobalHessianTracer, i::Integer) = collect(t.inputs[i])

"""
    tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GlobalGradientTracer`](@ref) and [`GlobalHessianTracer`](@ref) from input indices.
"""
function tracer(::Type{GlobalGradientTracer{I,S}}, index::Integer) where {I,S}
    return GlobalGradientTracer{I,S}(S(index))
end
function tracer(::Type{ConnectivityTracer{I,S}}, index::Integer) where {I,S}
    return ConnectivityTracer{I,S}(S(index))
end
function tracer(::Type{GlobalHessianTracer{I,S,D}}, index::Integer) where {I,S,D}
    return GlobalHessianTracer{I,S,D}(D(index => S()))
end

function tracer(::Type{GlobalGradientTracer{I,S}}, inds::NTuple{N,<:Integer}) where {I,S,N}
    return GlobalGradientTracer{I,S}(S(inds))
end
function tracer(::Type{ConnectivityTracer{I,S}}, inds::NTuple{N,<:Integer}) where {I,S,N}
    return ConnectivityTracer{I,S}(S(inds))
end
function tracer(
    ::Type{GlobalHessianTracer{I,S,D}}, inds::NTuple{N,<:Integer}
) where {I,S,D,N}
    return GlobalHessianTracer{I,S,D}(D(i => S() for i in inds))
end
