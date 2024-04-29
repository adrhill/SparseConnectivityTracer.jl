#==============#
# Connectivity #
#==============#

"""
    ConnectivityTracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`pattern`](@ref).
"""
struct ConnectivityTracer <: AbstractTracer
    inputs::BitSet # indices of connected, enumerated inputs
end

function Base.show(io::IO, t::ConnectivityTracer)
    return Base.show_delim_array(io, inputs(t), "ConnectivityTracer(", ',', ')', true)
end

const EMPTY_CONNECTIVITY_TRACER   = ConnectivityTracer(BitSet())
empty(::ConnectivityTracer)       = EMPTY_CONNECTIVITY_TRACER
empty(::Type{ConnectivityTracer}) = EMPTY_CONNECTIVITY_TRACER

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `ConnectivityTracer`.
# When this happens, we create a new empty tracer with no input pattern.
ConnectivityTracer(::Number) = EMPTY_CONNECTIVITY_TRACER
ConnectivityTracer(t::ConnectivityTracer) = t

## Unions of tracers
function uniontracer(a::ConnectivityTracer, b::ConnectivityTracer)
    return ConnectivityTracer(union(a.inputs, b.inputs))
end

#==========#
# Jacobian #
#==========#

"""
    JacobianTracer(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`pattern`](@ref).
"""
struct JacobianTracer <: AbstractTracer
    inputs::BitSet
end

function Base.show(io::IO, t::JacobianTracer)
    return Base.show_delim_array(io, inputs(t), "JacobianTracer(", ',', ')', true)
end

const EMPTY_JACOBIAN_TRACER   = JacobianTracer(BitSet())
empty(::JacobianTracer)       = EMPTY_JACOBIAN_TRACER
empty(::Type{JacobianTracer}) = EMPTY_JACOBIAN_TRACER

JacobianTracer(::Number) = EMPTY_JACOBIAN_TRACER
JacobianTracer(t::JacobianTracer) = t

## Unions of tracers
function uniontracer(a::JacobianTracer, b::JacobianTracer)
    return JacobianTracer(union(a.inputs, b.inputs))
end

#=========#
# Hessian #
#=========#
const HessianDict = Dict{UInt64,BitSet}
"""
    HessianTracer(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`pattern`](@ref).
"""
struct HessianTracer <: AbstractTracer
    inputs::HessianDict
end
function Base.show(io::IO, t::HessianTracer)
    println(io, "HessianTracer(")
    for key in keys(t.inputs)
        print(io, "  ", key, " => ")
        Base.show_delim_array(io, collect(t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

const EMPTY_HESSIAN_TRACER   = HessianTracer(HessianDict())
empty(::HessianTracer)       = EMPTY_HESSIAN_TRACER
empty(::Type{HessianTracer}) = EMPTY_HESSIAN_TRACER

HessianTracer(::Number) = empty(HessianTracer)
HessianTracer(t::HessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::HessianTracer)
    d = deepcopy(t.inputs)
    for k in keys(d)
        union!(d[k], k)
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

Return raw `UInt64` input indices of a [`ConnectivityTracer`](@ref) or [`JacobianTracer`](@ref)

## Example
```jldoctest
julia> t = tracer(ConnectivityTracer, 1, 2, 4)
ConnectivityTracer(1, 2, 4)

julia> inputs(t)
3-element Vector{Int64}:
 1
 2
 4
```
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::JacobianTracer) = collect(t.inputs)

"""
    tracer(JacobianTracer, index)
    tracer(JacobianTracer, indices)
    tracer(ConnectivityTracer, index)
    tracer(ConnectivityTracer, indices)

Convenience constructor for [`JacobianTracer`](@ref) [`ConnectivityTracer`](@ref) from input indices.
"""
tracer(::Type{JacobianTracer}, index::Integer) = JacobianTracer(BitSet(index))
tracer(::Type{ConnectivityTracer}, index::Integer) = ConnectivityTracer(BitSet(index))
function tracer(::Type{HessianTracer}, index::Integer)
    return HessianTracer(Dict{UInt64,BitSet}(index => BitSet()))
end

function tracer(::Type{JacobianTracer}, inds::NTuple{N,<:Integer}) where {N}
    return JacobianTracer(BitSet(inds))
end
function tracer(::Type{ConnectivityTracer}, inds::NTuple{N,<:Integer}) where {N}
    return ConnectivityTracer(BitSet(inds))
end
function tracer(::Type{HessianTracer}, inds::NTuple{N,<:Integer}) where {N}
    return HessianTracer(Dict{UInt64,BitSet}(i => BitSet() for i in inds))
end

tracer(::Type{T}, inds...) where {T<:AbstractTracer} = tracer(T, inds)
