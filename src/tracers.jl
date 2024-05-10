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
    GradientTracer{I,S}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`jacobian_pattern`](@ref).
"""
struct GradientTracer{I<:Integer,S} <: AbstractTracer
    inputs::S
end
function GradientTracer(inputs::S) where {S}
    I = eltype(S)
    return GradientTracer{I,S}(inputs)
end

function Base.show(io::IO, t::GradientTracer{I,S}) where {I,S}
    return Base.show_delim_array(
        io, convert.(Int, inputs(t)), "GradientTracer{$I,$S}(", ',', ')', true
    )
end

empty(::Type{GradientTracer{I,S}}) where {I<:Integer,S} = GradientTracer{I,S}(S())

GradientTracer{I,S}(::Number) where {I<:Integer,S} = empty(GradientTracer{I,S})
GradientTracer(t::GradientTracer) = t

## Unions of tracers
function uniontracer(a::GradientTracer{I,S}, b::GradientTracer{I,S}) where {I,S}
    return GradientTracer(union(a.inputs, b.inputs))
end

#=========#
# Hessian #
#=========#
"""
    HessianTracer{I,S,D}(indexset) <: Number

Number type keeping track of input indices of previous computations with non-zero first and second derivatives.

$SET_TYPE_MESSAGE

For a higher-level interface, refer to [`hessian_pattern`](@ref).
"""
struct HessianTracer{I<:Integer,S,D<:AbstractDict{I,S}} <: AbstractTracer
    inputs::D
end
function Base.show(io::IO, t::HessianTracer{I,S,D}) where {I,S,D}
    println(io, "$(eltype(t))(")
    for key in keys(t.inputs)
        print(io, "  ", Int(key), " => ")
        Base.show_delim_array(io, convert.(Int, t.inputs[key]), "(", ',', ')', true)
        println(io, ",")
    end
    return print(io, ")")
end

function empty(::Type{HessianTracer{I,S,D}}) where {I<:Integer,S,D<:AbstractDict{I,S}}
    return HessianTracer{I,S,D}(D())
end

HessianTracer{I,S,D}(::Number) where {I<:Integer,S,D} = empty(HessianTracer{I,S,D})
HessianTracer(t::HessianTracer) = t

# Turn first-order interactions into second-order interactions
function promote_order(t::HessianTracer{I,S}) where {I,S}
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
function distributive_merge(a::HessianTracer{I,S,D}, b::HessianTracer{I,S,D}) where {I,S,D}
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

Return input indices of a [`ConnectivityTracer`](@ref) or [`GradientTracer`](@ref)
"""
inputs(t::ConnectivityTracer) = collect(t.inputs)
inputs(t::GradientTracer) = collect(t.inputs)
inputs(t::HessianTracer, i::Integer) = collect(t.inputs[i])

"""
    tracer(T, index) where {T<:AbstractTracer}

Convenience constructor for [`ConnectivityTracer`](@ref), [`GradientTracer`](@ref) and [`HessianTracer`](@ref) from input indices.
"""
function tracer(::Type{GradientTracer{I,S}}, index::Integer) where {I,S}
    return GradientTracer{I,S}(S(index))
end
function tracer(::Type{ConnectivityTracer{I,S}}, index::Integer) where {I,S}
    return ConnectivityTracer{I,S}(S(index))
end
function tracer(::Type{HessianTracer{I,S,D}}, index::Integer) where {I,S,D}
    return HessianTracer{I,S,D}(D(index => S()))
end

function tracer(::Type{GradientTracer{I,S}}, inds::NTuple{N,<:Integer}) where {I,S,N}
    return GradientTracer{I,S}(S(inds))
end
function tracer(::Type{ConnectivityTracer{I,S}}, inds::NTuple{N,<:Integer}) where {I,S,N}
    return ConnectivityTracer{I,S}(S(inds))
end
function tracer(::Type{HessianTracer{I,S,D}}, inds::NTuple{N,<:Integer}) where {I,S,D,N}
    return HessianTracer{I,S,D}(D(i => S() for i in inds))
end
