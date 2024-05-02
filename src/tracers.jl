const AbstractIndexSet = AbstractSet{<:Integer}
abstract type AbstractTracer <: Number end

# Convenience constructor for empty tracers
empty(tracer::T) where {T<:AbstractTracer} = empty(T)

#==============#
# Connectivity #
#==============#

"""
    ConnectivityTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations.
The provided set type `S` has to be an `AbstractSet{<:Integer}`, e.g. `BitSet` or `Set{UInt64}`.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`connectivity_pattern`](@ref).
"""
struct ConnectivityTracer{S<:AbstractIndexSet} <: AbstractTracer
    inputs::S # indices of connected, enumerated inputs
end

function Base.show(io::IO, t::ConnectivityTracer{S}) where {S<:AbstractIndexSet}
    return Base.show_delim_array(io, inputs(t), "ConnectivityTracer{$S}(", ',', ')', true)
end

empty(::Type{ConnectivityTracer{S}}) where {S<:AbstractIndexSet} = ConnectivityTracer(S())

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
ConnectivityTracer{S}(::Number) where {S<:AbstractIndexSet} = empty(ConnectivityTracer{S})
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
function uniontracer(
    a::ConnectivityTracer{S}, b::ConnectivityTracer{S}
) where {S<:AbstractIndexSet}
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

#==========#
# Jacobian #
#==========#

"""
    JacobianTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.
The provided set type `S` has to be an `AbstractSet{<:Integer}`, e.g. `BitSet` or `Set{UInt64}`.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`jacobian_pattern`](@ref).
"""
struct JacobianTracer{S<:AbstractIndexSet} <: AbstractTracer
    inputs::S
end

function Base.show(io::IO, t::JacobianTracer{S}) where {S<:AbstractIndexSet}
    return Base.show_delim_array(io, inputs(t), "JacobianTracer{$S}(", ',', ')', true)
end

empty(::Type{JacobianTracer{S}}) where {S<:AbstractIndexSet} = JacobianTracer(S())

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

JacobianTracer{S}(::Number) where {S<:AbstractIndexSet} = empty(JacobianTracer{S})
JacobianTracer(t::JacobianTracer) = t

## Unions of tracers
function uniontracer(a::JacobianTracer{S}, b::JacobianTracer{S}) where {S<:AbstractIndexSet}
    return JacobianTracer(union(a.inputs, b.inputs))
end

#=========#
# Hessian #
#=========#
"""
    HessianTracer{S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.
The provided set type `S` has to be an `AbstractSet{<:Integer}`, e.g. `BitSet` or `Set{UInt64}`.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`hessian_pattern`](@ref).
"""
struct HessianTracer{S<:AbstractIndexSet} <: AbstractTracer
    inputs::Dict{UInt64,S}
end
function Base.show(io::IO, t::HessianTracer{S}) where {S<:AbstractIndexSet}
    println(io, "HessianTracer{", S, "}(")
    for key in keys(t.inputs)
        print(io, "  ", key, " => ")
        Base.show_delim_array(io, collect(t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

function empty(::Type{HessianTracer{S}}) where {S<:AbstractIndexSet}
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

HessianTracer{S}(::Number) where {S<:AbstractIndexSet} = empty(HessianTracer{S})
HessianTracer(t::HessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::HessianTracer)
    d = deepcopy(t.inputs)
    ks = keys(d)
    for v in values(d)
        union!(v, ks)
    end
    return HessianTracer(d)
end

# Merge first- and second-order terms in an "additive" fashion
function additive_merge(a::HessianTracer, b::HessianTracer)
    da = deepcopy(a.inputs)
    db = b.inputs
    for k in keys(db)
        if haskey(da, k)
            union!(da[k], db[k])
        else
            push!(da, k => db[k])
        end
    end
    return HessianTracer(da)
end

# Merge first- and second-order terms in a "distributive" fashion
function distributive_merge(a::HessianTracer, b::HessianTracer)
    da = deepcopy(a.inputs)
    db = deepcopy(b.inputs)
    for ka in keys(da)
        for kb in keys(db)
            # add second-order interaction term
            union!(da[ka], kb)
            union!(db[kb], ka)
        end
    end
    merge!(da, db)
    return HessianTracer(da)
end

#===========#
# Utilities #
#===========#

## Access inputs
"""
    inputs(tracer)

Return input indices of a [`ConnectivityTracer`](@ref) or [`JacobianTracer`](@ref)

## Example
```jldoctest
julia> a = tracer(ConnectivityTracer{BitSet}, 2)
ConnectivityTracer{BitSet}(2,)

julia> b = tracer(ConnectivityTracer{BitSet}, 4)
ConnectivityTracer{BitSet}(4,)

julia> c = a + b
ConnectivityTracer{BitSet}(2, 4)

julia> inputs(c)
2-element Vector{Int64}:
 2
 4
```
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::JacobianTracer) = collect(t.inputs)

"""
    tracer(ConnectivityTracer{S}, index)
    tracer(ConnectivityTracer{S}, indices)
    tracer(JacobianTracer{S}, index)
    tracer(JacobianTracer{S}, indices)
    tracer(HessianTracer{S}, index)
    tracer(HessianTracer{S}, indices)

Convenience constructor for [`ConnectivityTracer`](@ref), [`JacobianTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
The provided set type `S` has to be an `AbstractSet{<:Integer}`, e.g. `BitSet` or `Set{UInt64}`.

## Example
```jldoctest
julia> tracer(JacobianTracer{BitSet}, 2)
JacobianTracer{BitSet}(2,)

julia> tracer(HessianTracer{Set{UInt64}}, 2)
HessianTracer{Set{UInt64}}(
  2 => (),
)
```
"""
tracer(::Type{JacobianTracer{S}}, index::Integer) where {S<:AbstractIndexSet} =
    JacobianTracer(S(index))
function tracer(::Type{ConnectivityTracer{S}}, index::Integer) where {S<:AbstractIndexSet}
    return ConnectivityTracer(S(index))
end
function tracer(::Type{HessianTracer{S}}, index::Integer) where {S<:AbstractIndexSet}
    return HessianTracer(Dict{UInt64,S}(index => S()))
end

function tracer(
    ::Type{JacobianTracer{S}}, inds::NTuple{N,<:Integer}
) where {N,S<:AbstractIndexSet}
    return JacobianTracer{S}(S(inds))
end
function tracer(
    ::Type{ConnectivityTracer{S}}, inds::NTuple{N,<:Integer}
) where {N,S<:AbstractIndexSet}
    return ConnectivityTracer{S}(S(inds))
end
function tracer(
    ::Type{HessianTracer{S}}, inds::NTuple{N,<:Integer}
) where {N,S<:AbstractIndexSet}
    return HessianTracer{S}(Dict{UInt64,S}(i => S() for i in inds))
end
