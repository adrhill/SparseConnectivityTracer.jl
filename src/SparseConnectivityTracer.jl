module SparseConnectivityTracer
import Random: rand, AbstractRNG, SamplerType

include("scalar_ops.jl")

# Input connectivity tracer
struct Tracer <: Number
    inputs::Set{UInt64} # indices of connected, enumerated inputs
end

Tracer() = Tracer(Set{UInt64}())
Tracer(a::Tracer, b::Tracer) = Tracer(union(a.inputs, b.inputs))

# Get input indices as sorted Int array
inputs(t::Tracer) = sort(Int.(keys(t.inputs.dict)))

# Pretty printing
function Base.show(io::IO, t::Tracer)
    return Base.show_delim_array(io, inputs(t), "Tracer(", ',', ')', true)
end

# Enumerate inputs
tracer(index::Integer) = Tracer(Set{UInt64}(index)) # lower-case convenience constructor

trace(x) = trace(x, 1)
trace(::Number, i) = tracer(i)
function trace(x::AbstractArray, i)
    indices = (i - 1) .+ reshape(1:length(x), size(x))
    return tracer.(indices)
end

istracer(x) = false
istracer(x::Tracer) = true
istracer(x::AbstractArray{Tracer}) = true

# Extent core operators
for fn in (:+, :-, :*, :/)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    for T in (:Number,)
        @eval Base.$fn(t::Tracer, ::$T) = t
        @eval Base.$fn(::$T, t::Tracer) = t
    end
end

Base.:^(a::Tracer, b::Tracer) = Tracer(a, b)
for T in (:Number, :Integer, :Rational)
    @eval Base.:^(t::Tracer, ::$T) = t
    @eval Base.:^(::$T, t::Tracer) = t
end
Base.:^(t::Tracer, ::Irrational{:ℯ}) = t
Base.:^(::Irrational{:ℯ}, t::Tracer) = t

# Two-argument functions
for fn in (:div, :fld, :cld)
    @eval Base.$fn(a::Tracer, b::Tracer) = Tracer(a, b)
    @eval Base.$fn(t::Tracer, ::Number) = t
    @eval Base.$fn(::Number, t::Tracer) = t
end

# Single-argument functions
for fn in scalar_operations
    @eval Base.$fn(t::Tracer) = t
end

function connectivity(f, x)
    xt = inputtrace(x)
    yt = f(xt)
    n, m = length(xt), length(yt)

    # Construct connectivity matrix of size (ouput_dim, input_dim)
    C = BitArray(undef, m, n)
    for i in axes(C, 1)
        tracer = yt[i]
        for j in axes(C, 2)
            C[i, j] = j ∈ tracer.inputs
        end
    end
    return C
end

export Tracer, trace, inputs
export connectivity

end
