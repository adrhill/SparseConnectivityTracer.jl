"""
    Tracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).
For a higher-level interface, refer to [`connectivity`](@ref).

## Examples
By enumerating inputs with tracers, we can keep track of input connectivities:
```jldoctest
julia> xt = [tracer(1), tracer(2), tracer(3)]
3-element Vector{Tracer}:
 Tracer(1,)
 Tracer(2,)
 Tracer(3,)

julia> f(x) = [x[1]^2, 2 * x[1] * x[2]^2, sin(x[3])];

julia> yt = f(xt)
3-element Vector{Tracer}:
   Tracer(1,)
 Tracer(1, 2)
   Tracer(3,)
```

This works by overloading operators to either keep input connectivities constant, 
compute unions or set connectivities to zero:
```jldoctest Tracer
julia> x = tracer(1, 2, 3)
Tracer(1, 2, 3)

julia> sin(x)  # Most operators don't modify input connectivities.
Tracer(1, 2, 3)

julia> 2 * x^3
Tracer(1, 2, 3)

julia> zero(x) # Tracer is strictly operator overloading... 
Tracer()

julia> 0 * x   # ...and doesn't look at input values.
Tracer(1, 2, 3)

julia> y = tracer(3, 5)
Tracer(3, 5)

julia> x + y   # Operations on two Tracers construct union sets
Tracer(1, 2, 3, 5)

julia> x ^ y
Tracer(1, 2, 3, 5)
```

[`Tracer`](@ref) also supports random number generation and pre-allocations:
```jldoctest Tracer
julia> M = rand(Tracer, 3, 2)
3×2 Matrix{Tracer}:
 Tracer()  Tracer()
 Tracer()  Tracer()
 Tracer()  Tracer()

julia> similar(M)
3×2 Matrix{Tracer}:
 Tracer()  Tracer()
 Tracer()  Tracer()
 Tracer()  Tracer()

julia> M * [x, y]
3-element Vector{Tracer}:
 Tracer(1, 2, 3, 5)
 Tracer(1, 2, 3, 5)
 Tracer(1, 2, 3, 5)
```
"""
struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

const EMPTY_TRACER = Tracer(Set{UInt64}())

emptytracer() = EMPTY_TRACER
uniontracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `Tracer`.
# When this happens, we create a new empty tracer with no input connectivity.
Tracer(::Number)  = emptytracer()
Tracer(t::Tracer) = t

"""
    tracer(index)
    tracer(indices)

Convenience constructor for [`Tracer`](@ref) from input indices.
"""
tracer(index::Integer) = Tracer(Set{UInt64}(index))
tracer(inds::NTuple{N,<:Integer}) where {N} = Tracer(Set{UInt64}(inds))
tracer(inds...)                             = tracer(inds)

# Utilities for accessing input indices
"""
    inputs(tracer)

Return raw `UInt64` input indices of a [`Tracer`](@ref).
See also [`sortedinputs`](@ref).

## Example
```jldoctest
julia> t = tracer(1, 2, 4)
Tracer(1, 2, 4)

julia> inputs(t)
3-element Vector{UInt64}:
 0x0000000000000004
 0x0000000000000002
 0x0000000000000001
```
"""
inputs(t::Tracer) = collect(keys(t.inputs.dict))

"""
    sortedinputs(tracer)
    sortedinputs([T=Int], tracer)

Return sorted input indices of a [`Tracer`](@ref).
See also [`inputs`](@ref).

## Example
```jldoctest
julia> t = tracer(1, 2, 4)
Tracer(1, 2, 4)

julia> sortedinputs(t)
3-element Vector{Int64}:
 1
 2
 4

julia> sortedinputs(UInt8, t)
3-element Vector{UInt8}:
 0x01
 0x02
 0x04
```
"""
sortedinputs(t::Tracer) = sortedinputs(Int, t)
sortedinputs(T::Type, t::Tracer) = convert.(T, sort!(inputs(t)))

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, sortedinputs(Int, t), "Tracer(", ',', ')', true)
end
