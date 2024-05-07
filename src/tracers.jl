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
    ConnectivityTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`connectivity_pattern`](@ref).
"""
struct ConnectivityTracer{S} <: AbstractTracer
    inputs::S # indices of connected, enumerated inputs
end

function Base.show(io::IO, t::ConnectivityTracer{S}) where {S}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "ConnectivityTracer{$S}(", ',', ')', true
    )
end

empty(::Type{ConnectivityTracer{S}}) where {S} = ConnectivityTracer(S())

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer{S}(::Number) where {S} = empty(ConnectivityTracer{S})
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
function uniontracer(a::ConnectivityTracer{S}, b::ConnectivityTracer{S}) where {S}
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

#==========#
# Jacobian #
#==========#

"""
    JacobianTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`jacobian_pattern`](@ref).
"""
struct JacobianTracer{S} <: AbstractTracer
    inputs::S
end

function Base.show(io::IO, t::JacobianTracer{S}) where {S}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "JacobianTracer{$S}(", ',', ')', true
    )
end

empty(::Type{JacobianTracer{S}}) where {S} = JacobianTracer(S())

JacobianTracer{S}(::Number) where {S} = empty(JacobianTracer{S})
JacobianTracer(t::JacobianTracer) = t

## Unions of tracers
function uniontracer(a::JacobianTracer{S}, b::JacobianTracer{S}) where {S}
    return JacobianTracer(union(a.inputs, b.inputs))
end

#=========#
# Hessian #
#=========#
"""
    HessianTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`hessian_pattern`](@ref).
"""
struct HessianTracer{S,I<:Integer} <: AbstractTracer
    inputs::Dict{I,S}
end
function Base.show(io::IO, t::HessianTracer{S}) where {S}
    println(io, "HessianTracer{", S, "}(")
    for key in keys(t.inputs)
        print(io, "  ", Int(key), " => ")
        Base.show_delim_array(io, convert.(Int, t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

function empty(::Type{HessianTracer{S,I}}) where {S,I}
    return HessianTracer(Dict{I,S}())
end

HessianTracer{S,I}(::Number) where {S,I} = empty(HessianTracer{S,I})
HessianTracer(t::HessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::HessianTracer{S}) where {S}
    d = deepcopy(t.inputs)
    s = keys2set(S, d)
    for (k, v) in pairs(d)
        d[k] = union(v, s)  # ignores symmetry
    end
    return HessianTracer(d)
end

# Merge first- and second-order terms in an "additive" fashion
function additive_merge(a::HessianTracer, b::HessianTracer)
    return HessianTracer(mergewith(union, a.inputs, b.inputs))
end

# Merge first- and second-order terms in a "distributive" fashion
function distributive_merge(a::HessianTracer{S}, b::HessianTracer{S}) where {S}
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
    return HessianTracer(merge(da, db))
end

#===========#
# Utilities #
#===========#

## Access inputs
"""
    inputs(tracer)

Return input indices of a [`ConnectivityTracer`](@ref) or [`JacobianTracer`](@ref)
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::JacobianTracer) = collect(t.inputs)
inputs(t::HessianTracer, i::Integer) = collect(t.inputs[i])

"""
    tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`JacobianTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
tracer(::Type{JacobianTracer{S}}, index::Integer) where {S} = JacobianTracer(S(index))
function tracer(::Type{ConnectivityTracer{S}}, index::Integer) where {S}
    return ConnectivityTracer(S(index))
end
function tracer(::Type{HessianTracer{S}}, index::Integer) where {S}
    I = eltype(S)
    return HessianTracer{S,I}(Dict{I,S}(index => S()))
end

function tracer(::Type{JacobianTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    return JacobianTracer{S}(S(inds))
end
function tracer(::Type{ConnectivityTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    return ConnectivityTracer{S}(S(inds))
end
function tracer(::Type{HessianTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    I = eltype(S)
    return HessianTracer{S,I}(Dict{I,S}(i => S() for i in inds))
end
