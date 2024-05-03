abstract type AbstractTracer <: Number end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)

#==============#
# Connectivity #
#==============#

const SET_TYPE_MESSAGE = """
The provided index set type `S` has to satisfy the following conditions:

- it is an iterable with `<:Integer` element type
- it implements `union`

Subtypes of `AbstractSet{<:Integer}` are a natural choice, like `BitSet` or `Set{UInt64}`.
"""

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
    return Base.show_delim_array(io, inputs(t), "ConnectivityTracer{$S}(", ',', ')', true)
end

empty(::Type{ConnectivityTracer{S}}) where {S} = ConnectivityTracer(S())

# Performance can be gained by not re-allocating empty tracers
const EMPTY_CONNECTIVITY_TRACER_BITSET     = ConnectivityTracer(BitSet())
const EMPTY_CONNECTIVITY_TRACER_SET_UINT8  = ConnectivityTracer(Set{UInt8}())
const EMPTY_CONNECTIVITY_TRACER_SET_UINT16 = ConnectivityTracer(Set{UInt16}())
const EMPTY_CONNECTIVITY_TRACER_SET_UINT32 = ConnectivityTracer(Set{UInt32}())
const EMPTY_CONNECTIVITY_TRACER_SET_UINT64 = ConnectivityTracer(Set{UInt64}())

empty(::Type{ConnectivityTracer{BitSet}})      = EMPTY_CONNECTIVITY_TRACER_BITSET
empty(::Type{ConnectivityTracer{Set{UInt8}}})  = EMPTY_CONNECTIVITY_TRACER_SET_UINT8
empty(::Type{ConnectivityTracer{Set{UInt16}}}) = EMPTY_CONNECTIVITY_TRACER_SET_UINT16
empty(::Type{ConnectivityTracer{Set{UInt32}}}) = EMPTY_CONNECTIVITY_TRACER_SET_UINT32
empty(::Type{ConnectivityTracer{Set{UInt64}}}) = EMPTY_CONNECTIVITY_TRACER_SET_UINT64

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
    return Base.show_delim_array(io, inputs(t), "JacobianTracer{$S}(", ',', ')', true)
end

empty(::Type{JacobianTracer{S}}) where {S} = JacobianTracer(S())

# Performance can be gained by not re-allocating empty tracers
const EMPTY_JACOBIAN_TRACER_BITSET     = JacobianTracer(BitSet())
const EMPTY_JACOBIAN_TRACER_SET_UINT8  = JacobianTracer(Set{UInt8}())
const EMPTY_JACOBIAN_TRACER_SET_UINT16 = JacobianTracer(Set{UInt16}())
const EMPTY_JACOBIAN_TRACER_SET_UINT32 = JacobianTracer(Set{UInt32}())
const EMPTY_JACOBIAN_TRACER_SET_UINT64 = JacobianTracer(Set{UInt64}())

empty(::Type{JacobianTracer{BitSet}})      = EMPTY_JACOBIAN_TRACER_BITSET
empty(::Type{JacobianTracer{Set{UInt8}}})  = EMPTY_JACOBIAN_TRACER_SET_UINT8
empty(::Type{JacobianTracer{Set{UInt16}}}) = EMPTY_JACOBIAN_TRACER_SET_UINT16
empty(::Type{JacobianTracer{Set{UInt32}}}) = EMPTY_JACOBIAN_TRACER_SET_UINT32
empty(::Type{JacobianTracer{Set{UInt64}}}) = EMPTY_JACOBIAN_TRACER_SET_UINT64

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
struct HessianTracer{S} <: AbstractTracer
    inputs::Dict{UInt64,S}
end
function Base.show(io::IO, t::HessianTracer{S}) where {S}
    println(io, "HessianTracer{", S, "}(")
    for key in keys(t.inputs)
        print(io, "  ", key, " => ")
        Base.show_delim_array(io, collect(t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

function empty(::Type{HessianTracer{S}}) where {S}
    return HessianTracer(Dict{UInt64,S}())
end

# Performance can be gained by not re-allocating empty tracers
const EMPTY_HESSIAN_TRACER_BITSET     = HessianTracer(Dict{UInt64,BitSet}())
const EMPTY_HESSIAN_TRACER_SET_UINT8  = HessianTracer(Dict{UInt64,Set{UInt8}}())
const EMPTY_HESSIAN_TRACER_SET_UINT16 = HessianTracer(Dict{UInt64,Set{UInt16}}())
const EMPTY_HESSIAN_TRACER_SET_UINT32 = HessianTracer(Dict{UInt64,Set{UInt32}}())
const EMPTY_HESSIAN_TRACER_SET_UINT64 = HessianTracer(Dict{UInt64,Set{UInt64}}())

empty(::Type{HessianTracer{BitSet}})      = EMPTY_HESSIAN_TRACER_BITSET
empty(::Type{HessianTracer{Set{UInt8}}})  = EMPTY_HESSIAN_TRACER_SET_UINT8
empty(::Type{HessianTracer{Set{UInt16}}}) = EMPTY_HESSIAN_TRACER_SET_UINT16
empty(::Type{HessianTracer{Set{UInt32}}}) = EMPTY_HESSIAN_TRACER_SET_UINT32
empty(::Type{HessianTracer{Set{UInt64}}}) = EMPTY_HESSIAN_TRACER_SET_UINT64

HessianTracer{S}(::Number) where {S} = empty(HessianTracer{S})
HessianTracer(t::HessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::HessianTracer)
    d = deepcopy(t.inputs)
    for (k, v) in pairs(d)
        d[k] = union(v, keys(d))  # works by not being clever with symmetry
    end
    return HessianTracer(d)
end

# Merge first- and second-order terms in an "additive" fashion
function additive_merge(a::HessianTracer, b::HessianTracer)
    return HessianTracer(mergewith(union, a.inputs, b.inputs))
end

# Merge first- and second-order terms in a "distributive" fashion
function distributive_merge(a::HessianTracer, b::HessianTracer)
    da = deepcopy(a.inputs)
    db = deepcopy(b.inputs)
    # add second-order interaction term, works by not being clever with symmetry
    for (ka, va) in pairs(da)
        da[ka] = union(va, keys(db))
    end
    for (kb, vb) in pairs(db)
        da[kb] = union(vb, keys(da))
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

"""
    tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`JacobianTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
tracer(::Type{JacobianTracer{S}}, index::Integer) where {S} = JacobianTracer(S(index))
function tracer(::Type{ConnectivityTracer{S}}, index::Integer) where {S}
    return ConnectivityTracer(S(index))
end
function tracer(::Type{HessianTracer{S}}, index::Integer) where {S}
    return HessianTracer(Dict{UInt64,S}(index => S()))
end

function tracer(::Type{JacobianTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    return JacobianTracer{S}(S(inds))
end
function tracer(::Type{ConnectivityTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    return ConnectivityTracer{S}(S(inds))
end
function tracer(::Type{HessianTracer{S}}, inds::NTuple{N,<:Integer}) where {N,S}
    return HessianTracer{S}(Dict{UInt64,S}(i => S() for i in inds))
end
