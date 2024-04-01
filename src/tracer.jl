"""
    Tracer(indexset) <: Number

Number type keeping track of input indices of previous computations.

See also the convenience constructor [`tracer`](@ref).

## Examples
```julia-repl
julia> x = tracer(1, 2, 3)
Tracer(1, 2, 3)

julia> sin(x)
Tracer(1, 2, 3)

julia> 2 * x^3
Tracer(1, 2, 3)

julia> 0 * x   # Note: Tracer is strictly operator overloading...
Tracer(1, 2, 3)

julia> zero(x) # ...this can be overloaded
Tracer()

julia> y = tracer(3, 5)
Tracer(3, 5)

julia> x + y
Tracer(1, 2, 3, 5)

julia> x ^ y
Tracer(1, 2, 3, 5)

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

# We have to be careful when defining constructors:
# Generic code expecting "regular" numbers `x` will sometimes convert them 
# by calling `T(x)` (instead of `convert(T, x)`), where `T` can be `Tracer`.
# When this happens, we create a new empty tracer with no input connectivity.
Tracer(::Number)  = tracer()
Tracer(t::Tracer) = t
# We therefore exclusively use the lower-case `tracer` for convenience constructors

"""
    tracer(index)
    tracer(indices)

Convenience constructor for [`Tracer`](@ref) from input indices.
"""
tracer() = Tracer(Set{UInt64}())
tracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

tracer(index::Integer)                      = Tracer(Set{UInt64}(index))
tracer(inds::NTuple{N,<:Integer}) where {N} = Tracer(Set{UInt64}(inds))
tracer(inds...)                             = tracer(inds)

# Utilities for accessing input indices
"""
    inputs(tracer)

Return raw `UInt64` input indices of a [`Tracer`](@ref).
"""
inputs(t::Tracer) = collect(keys(t.inputs.dict))

"""
    sortedinputs(tracer)
    sortedinputs([T=Int], tracer)

Return sorted input indices of a [`Tracer`](@ref).
"""
sortedinputs(t::Tracer) = sortedinputs(Int, t)
sortedinputs(T::Type, t::Tracer) = convert.(T, sort!(inputs(t)))

function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, sortedinputs(Int, t), "Tracer(", ',', ')', true)
end
